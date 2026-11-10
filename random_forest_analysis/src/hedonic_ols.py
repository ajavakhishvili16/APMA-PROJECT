"""Estimate the hedonic pricing model by OLS on the training set.

    python src/hedonic_ols.py

Model:  ln(P/A) = a + exposure + medium + support + sale season
                  + b_F * fame + b1 * ln A + b2 * (ln A)^2 + e
Standard errors are clustered by artist (column `cluster`).
pct_effect_kennedy = 100 * (exp(b - Var(b)/2) - 1), Kennedy (1981), for dummies.

Writes results/ols_coefficients.csv and results/ols_summary.txt.
"""
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import FORMULA, RESULTS, TRAIN, load


def tidy_name(term):
    """C(exposure, Treatment('1-lot'))[T.local] -> 'exposure: local'."""
    return re.sub(r"C\((\w+), Treatment\('[^']+'\)\)\[T\.(.+)\]", r"\1: \2", term)


def fit(d):
    groups = pd.factorize(d.cluster)[0]
    return smf.ols(FORMULA, d).fit(cov_type="cluster", cov_kwds={"groups": groups})


def coefficient_table(m):
    ci = m.conf_int()
    t = pd.DataFrame({
        "coef": m.params, "se_cluster": m.bse, "t": m.tvalues, "p": m.pvalues,
        "ci_low": ci[0], "ci_high": ci[1],
    })
    dummy = t.index.str.startswith("C(")
    t["pct_effect_kennedy"] = np.where(
        dummy, 100 * (np.exp(t.coef - 0.5 * t.se_cluster ** 2) - 1), np.nan)
    t.index = [tidy_name(i) for i in t.index]
    return t


if __name__ == "__main__":
    train = load(TRAIN)
    m = fit(train)
    table = coefficient_table(m)
    table.to_csv(RESULTS / "ols_coefficients.csv")
    (RESULTS / "ols_summary.txt").write_text(str(m.summary()))
    print(f"n = {int(m.nobs)}, parameters = {len(m.params)}, R^2 = {m.rsquared:.4f}")
    print(table[["coef", "se_cluster", "p", "pct_effect_kennedy"]].round(4).to_string())
    print(f"\nwrote {RESULTS / 'ols_coefficients.csv'}")
