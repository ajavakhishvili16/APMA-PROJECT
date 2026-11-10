import time, pandas as pd, numpy as np
from numbers_parser import Document

# CPI-U, U.S. city average, all items, not seasonally adjusted (1982-84=100), BLS
# (via Idaho Dept. of Labor 2026 CPI table). Oct 2025 not published (2025 federal
# funding lapse) -> average of Sep and Nov 2025.
CPI = {"2023-10":307.671,"2023-11":307.051,"2023-12":306.746,
"2024-01":308.417,"2024-02":310.326,"2024-03":312.332,"2024-04":313.548,"2024-05":314.069,
"2024-06":314.175,"2024-07":314.540,"2024-08":314.796,"2024-09":315.301,"2024-10":315.664,
"2024-11":315.493,"2024-12":315.605,"2025-01":317.671,"2025-02":319.082,"2025-03":319.799,
"2025-04":320.795,"2025-05":321.465,"2025-06":322.561,"2025-07":323.048,"2025-08":323.976,
"2025-09":324.800,"2025-10":(324.800+324.122)/2,"2025-11":324.122,"2025-12":324.054,
"2026-01":325.252,"2026-02":326.785,"2026-03":330.213,"2026-04":333.020,"2026-05":335.123,
"2026-06":333.952,"2026-07":333.918,
"2026-08":334.980,"2026-09":336.940}   # Aug-Sep 2026: BLS release USDL-26-1574
BASE_MONTH = "2026-09"                       # latest month published (BLS Sep 2026 CPI release)
BASE = CPI[BASE_MONTH]

t0=time.time()
doc = Document("in.numbers"); tab = doc.sheets[0].tables[0]
rows = tab.rows(values_only=True)
hdr = rows[0]; df = pd.DataFrame(rows[1:], columns=hdr)
col = {h:i for i,h in enumerate(hdr)}
print("rows", len(df), "read in", round(time.time()-t0), "s")

# if no artist --> exposure: 1-lot
noname = df["artist_name"].isna() | (df["artist_name"].astype(str).str.strip() == "")
print("lots with no artist name:", int(noname.sum()))
for r in np.where(noname)[0]:
    for c, v in [("exposure", "1-lot"), ("exposure_full", "1-lot"), ("fame", 0.0), ("wiki_found", 0.0)]:
        tab.write(r + 1, col[c], v)

# adjust inflation inflation: real 2026 USD = nominal USD * CPI(base) / CPI(sale month)
month = pd.to_datetime(df["auction_date"]).dt.strftime("%Y-%m")
month_capped = month.where(month <= BASE_MONTH, BASE_MONTH)   # Aug-Oct 2026: no CPI yet -> factor 1
cpi_m = month_capped.map(CPI)
assert cpi_m.notna().all(), month[cpi_m.isna()].unique()
factor = BASE / cpi_m
new = {"cpi_sale_month": cpi_m, "inflation_factor": factor,
       "price_usd_2026": df["price_usd"].astype(float) * factor,
       "estimate_low_usd_2026": df["estimate_low_usd"].astype(float) * factor,
       "estimate_high_usd_2026": df["estimate_high_usd"].astype(float) * factor}
start = len(hdr)
for j, (name, s) in enumerate(new.items()):
    c = start + j
    tab.write(0, c, name)
    for r, v in enumerate(s.values):
        if pd.notna(v): tab.write(r + 1, c, float(v))
print("factor range", round(factor.min(),4), "to", round(factor.max(),4))
print(pd.DataFrame({"month":month,"factor":factor}).groupby(month.str[:4]).factor.agg(["min","max"]).round(4))
doc.save("Dataset_inflation_adjusted_sep2026.numbers")
df2 = df.assign(**new); df2.loc[noname, ["exposure","exposure_full"]] = "1-lot"; df2.loc[noname, ["fame","wiki_found"]] = 0
df2.to_csv("Dataset_inflation_adjusted_sep2026_check.csv", index=False)
print("saved in", round(time.time()-t0), "s")
print(df2.loc[[0,1], ["artist_name","auction_date","price_usd","inflation_factor","price_usd_2026"]])
print(df2[pd.to_datetime(df2.auction_date)<"2024-03-01"].head(2)[["artist_name","auction_date","currency","price_usd","inflation_factor","price_usd_2026"]])
