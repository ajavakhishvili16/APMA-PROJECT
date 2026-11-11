"""
Jana: Estimation for coefficient and joint wald test. 

Adjustments from previous data models as well
split data into two

current model: OLS, standard erros clustered by artists

  ln(P/A) = a + Exposure + b_f*Fame + b1*lnA + b2*(lnA)^2 + Medium + Support + SaleSeason(Q1-Q4, base Q1) + CreationYear(period dummies, base 1945-69) + e

  P = price in USD, A = area in cm^2, Fame = ln(1 + Wikipedia language editions)

"""
import os, sys, re, urllib.request
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# data: pass a path, or the script uses dataset_inflation_adjusted.csv and downloads it if missing
DATA_FILE = "dataset_inflation_adjusted.csv"
DATA_URL = "https://raw.githubusercontent.com/OWNER/REPO/main/dataset_inflation_adjusted.csv"  # TODO: repo raw link
PATH = sys.argv[1] if len(sys.argv) > 1 else DATA_FILE
if not os.path.exists(PATH):
    print(f"{PATH} not found, downloading from {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, PATH)
PRICE = "price_usd_2026"          # inflation-adjusted price (September 2026 USD)
SPLIT_DATE = "2026-01-01"


d = pd.read_csv(PATH, low_memory=False)
d["auction_date"] = pd.to_datetime(d["auction_date"])
d = d[d[PRICE].gt(0) & d["area_cm2"].gt(0)].copy()
# lots with no artist name: exposure '1-lot', fame 0 (set in the data); each such lot
# is its own cluster, since it cannot be linked to other sales
d["cluster"] = d["artist_key"].fillna(pd.Series("nameless_" + d.index.astype(str), index=d.index))

# variables
# outcome: log price per cm^2
d["ln_area"] = np.log(d["area_cm2"])
d["ln_area2"] = d["ln_area"] ** 2
d["ln_ppa"] = np.log(d[PRICE]) - d["ln_area"]

# medium and support from the medium text
m = d["medium"].fillna("").str.lower()
has = lambda p: m.str.contains(p, regex=True)
d["medium_c"] = np.select(
    [has(r"\boil\b") & ~has(r"oil pastel|oil stick"), has(r"acrylic"),
     has(r"mixed media|collage|assemblage"), has(r"watercol|gouache|tempera"),
     has(r"\bink\b"), has(r"pencil|graphite|charcoal|pastel|crayon|chalk|\bpen\b|cont[eé]")],
    ["oil", "acrylic", "mixed", "watercolour", "ink", "drawing"], "other")
d["support_c"] = np.select(
    [has(r"canvas|linen"), has(r"panel|board|masonite|wood|copper|card"),
     has(r"paper|silk|scroll|album")],
    ["canvas", "panel", "paper"], "other")

# date created: linear in year (centred at 1950, in decades), plus a dummy for unknown
d["created_unknown"] = d["artwork_year"].isna().astype(int)
d["year_created"] = ((d["artwork_year"] - 1950) / 10).fillna(0)   # decades from 1950

# sale quarter
d["quarter"] = d["auction_date"].dt.to_period("Q").astype(str)
# season = quarter of the YEAR (Q1-Q4), shared across years, so future sales get a coefficient
d["season"] = "Q" + d["auction_date"].dt.quarter.astype(str)
# year created as periods (a straight line had no effect), missing dates = 'unknown'
d["created_period"] = pd.cut(d["artwork_year"], [0, 1799, 1899, 1944, 1969, 1999, 2100],
    labels=["pre1800", "1800s", "1900_44", "1945_69", "1970_99", "2000plus"]
    ).astype(object).fillna("unknown").astype(str)

# split
train = d[d["auction_date"] < SPLIT_DATE].reset_index(drop=True)
test = d[d["auction_date"] >= SPLIT_DATE].reset_index(drop=True)
test.to_csv("test_set_2026.csv", index=False)
print(f"Training: {len(train):,} sales, {train['cluster'].nunique():,} clusters "
      f"({train['auction_date'].min():%Y-%m-%d} to {train['auction_date'].max():%Y-%m-%d})")
print(f"Test (2026): {len(test):,} sales, saved to test_set_2026.csv\n")

# model
BLOCKS = {
    "Exposure":     "C(exposure, Treatment('1-lot'))",
    "Fame":         "fame",
    "Size":         "ln_area + ln_area2",
    "Medium":       "C(medium_c, Treatment('oil'))",
    "Support":      "C(support_c, Treatment('canvas'))",
    "Sale season":  "C(season, Treatment('Q1'))",     # Q2, Q3, Q4 vs Q1
    "Year created": "C(created_period, Treatment('1945_69'))",
}

def fit(blocks, data=train):
    formula = "ln_ppa ~ " + " + ".join(blocks.values())
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(data["cluster"])[0]})

full = fit(BLOCKS)

def block_columns(term):
    """names of the coefficients belonging to one block"""
    if term.startswith("C("):
        return [c for c in full.params.index if c.startswith(term.split(",")[0].split(")")[0])]
    return [t.strip() for t in term.split("+")]

# coefficients
se = full.bse
coef = pd.DataFrame({
    "coef": full.params, "se_cluster": se, "t": full.tvalues, "p": full.pvalues,
    "ci_low": full.params - 1.96 * se, "ci_high": full.params + 1.96 * se,
    # percentage effect for dummies, Kennedy (1981) correction: 100*[exp(b - SE^2/2) - 1]
    "pct_effect_kennedy": 100 * (np.exp(full.params - se ** 2 / 2) - 1),
})
def clean(n):
    hit = re.match(r"C\((\w+).*\)\[T\.(.*)\]", n)
    return f"{hit.group(1)}: {hit.group(2)}" if hit else n
coef.index = [clean(i) for i in coef.index]
for c in ["fame", "ln_area", "ln_area2", "Intercept"]:
    coef.loc[c, "pct_effect_kennedy"] = np.nan       # not a dummy: % effect not meaningful
coef.to_csv("coefficients.csv")

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
print("COEFFICIENTS (outcome: ln price per cm^2, September-2026 USD; SEs clustered by artist)")
print(coef.to_string(), "\n")
print(f"n = {int(full.nobs):,}   clusters = {train['cluster'].nunique():,}   "
      f"R2 = {full.rsquared:.3f}   adj. R2 = {full.rsquared_adj:.3f}\n")

# joint Wald tests
rows = []
names = list(full.params.index)
for block, term in BLOCKS.items():
    cols = block_columns(term)
    R = np.zeros((len(cols), len(names)))
    for i, c in enumerate(cols):
        R[i, names.index(c)] = 1                   # H0: R beta = 0
    # W = (R b)' (R V R')^-1 (R b) with the clustered V;  F = W / q ~ F(q, G - 1)
    b = R @ full.params.values
    V = full.cov_params().values
    W = float(b @ np.linalg.solve(R @ V @ R.T, b))
    q = len(cols)
    G = train["cluster"].nunique()
    F = W / q
    reduced = fit({k: v for k, v in BLOCKS.items() if k != block})
    rows.append({"block": block, "q (coefs)": q, "W (chi2)": W, "F = W/q": F,
                 "p_value": float(stats.f.sf(F, q, G - 1)),
                 "dR2": full.rsquared - reduced.rsquared,
                 "dAdjR2": full.rsquared_adj - reduced.rsquared_adj,
                 "dAIC": reduced.aic - full.aic, "dBIC": reduced.bic - full.bic})
wald = pd.DataFrame(rows)
wald.to_csv("wald_tests.csv", index=False)
print("JOINT WALD TESTS  (H0: all coefficients in the block = 0; clustered covariance)")
print("dR2/dAIC/dBIC = how much worse the model gets when the block is removed")
print(wald.to_string(index=False), "\n")

# size elasticity
b1, b2 = full.params["ln_area"], full.params["ln_area2"]
print("SIZE: effect on ln(price per cm^2) = b1 + 2*b2*ln(A)")
for a_cm2 in [500, 2000, 10000]:
    e = b1 + 2 * b2 * np.log(a_cm2)
    print(f"  A = {a_cm2:>6} cm^2: per-cm^2 elasticity {e:+.3f}, total-price elasticity {1 + e:+.3f}")
if b2 != 0:
    print(f"  turning point: A* = exp(-b1 / (2*b2)) = {np.exp(-b1 / (2 * b2)):,.0f} cm^2 "
          f"(data range {train['area_cm2'].min():,.0f} to {train['area_cm2'].max():,.0f})")
