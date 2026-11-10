"""Technical nugget: random forest vs the hedonic OLS model.

    python src/random_forest.py

1. Fit the hedonic model by OLS on the training set (2023-2025).
2. Tune a random forest inside the training set only: fit on sales up to
   June 2025, validate on July-December 2025, pick the best settings.
3. Refit the forest on the full training set with those settings.
4. Predict the 2026 test set with both models and compare.

Both models get the same seven characteristics (exposure, fame, area, medium,
support, sale season, creation period) and predict the same target, y = ln(P/A).

Writes to results/:
  rf_vs_ols_metrics.json    accuracy table, bootstrap interval, extras
  rf_tuning.csv             validation R^2 for every setting tried
  rf_importance.csv         permutation importance (test set), forest and OLS
  rf_partial_dependence.png forest vs OLS shape for area and fame
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from common import BASE, RESULTS, SEED, TEST, TRAIN, load, metrics
from hedonic_ols import fit as fit_ols

N_TREES = 500
GRID_LEAF = [1, 5, 10, 20, 40]     # minimum sales per final box
GRID_MAX_FEATURES = [0.33, 0.5, 1.0]  # share of inputs tried at each split


def forest_inputs(d, levels):
    """Fame, ln A and one 0/1 indicator per category level (trees need no base level)."""
    X = pd.DataFrame({"fame": d.fame.values, "ln_area": d.ln_area.values})
    for col, lv in levels.items():
        for v in lv:
            X[f"{col}={v}"] = (d[col].astype(str) == v).values.astype(float)
    return X


def tune(train, levels):
    fit_part = train[train.date < "2025-07-01"]
    valid = train[train.date >= "2025-07-01"]
    rows = []
    for leaf in GRID_LEAF:
        for mf in GRID_MAX_FEATURES:
            rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=leaf, max_features=mf,
                                       n_jobs=-1, random_state=SEED)
            rf.fit(forest_inputs(fit_part, levels), fit_part.y)
            r2 = metrics(valid.y, rf.predict(forest_inputs(valid, levels)))["R2"]
            rows.append({"min_samples_leaf": leaf, "max_features": mf, "valid_R2": r2})
            print(f"  leaf {leaf:>2}, max_features {mf:<4}: validation R^2 {r2:.4f}")
    return pd.DataFrame(rows).sort_values("valid_R2", ascending=False)


def bootstrap_r2_gain(y, p_ols, p_rf, reps=2000):
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(reps):
        i = rng.integers(0, len(y), len(y))
        sst = ((y[i] - y[i].mean()) ** 2).sum()
        out.append(((y[i] - p_ols[i]) ** 2).sum() / sst - ((y[i] - p_rf[i]) ** 2).sum() / sst)
    return np.percentile(out, [2.5, 97.5]).tolist()


def permutation_importance(predict, d, groups, y, reps=5):
    """Rise in test MSE when one characteristic's columns are shuffled."""
    rng = np.random.default_rng(SEED)
    base = ((y - predict(d)) ** 2).mean()
    out = {}
    for name, cols in groups.items():
        rise = []
        for _ in range(reps):
            dp = d.copy()
            perm = rng.permutation(len(dp))
            dp[cols] = dp[cols].values[perm]
            rise.append(((y - predict(dp)) ** 2).mean() - base)
        out[name] = float(np.mean(rise))
    return out


def partial_dependence(rf, X_sample, col, grid):
    out = []
    for g in grid:
        Xg = X_sample.copy()
        Xg[col] = g
        out.append(rf.predict(Xg).mean())
    return np.array(out)


if __name__ == "__main__":
    train, test = load(TRAIN), load(TEST)
    levels = {c: sorted(train[c].astype(str).unique()) for c in BASE}
    print(f"training sales: {len(train)} ({train.date.min():%b %Y} - {train.date.max():%b %Y})")
    print(f"test sales:     {len(test)} ({test.date.min():%b %Y} - {test.date.max():%b %Y})")

    # 1. Hedonic OLS
    ols = fit_ols(train)

    # 2-3. Random forest
    print("\nTuning the forest inside the training set:")
    tuning = tune(train, levels)
    tuning.to_csv(RESULTS / "rf_tuning.csv", index=False)
    best = tuning.iloc[0]
    rf = RandomForestRegressor(n_estimators=N_TREES, min_samples_leaf=int(best.min_samples_leaf),
                               max_features=float(best.max_features), n_jobs=-1,
                               random_state=SEED, oob_score=True)
    X_train = forest_inputs(train, levels)
    rf.fit(X_train, train.y)

    # 4. Compare on 2026
    y = test.y.values
    p_ols = ols.predict(test).values
    p_rf = rf.predict(forest_inputs(test, levels))
    p_mean = np.full(len(y), train.y.mean())
    res = {
        "n_train": len(train), "n_test": len(test),
        "forest_settings": {"trees": N_TREES, "min_samples_leaf": int(best.min_samples_leaf),
                            "max_features": float(best.max_features), "inputs": X_train.shape[1]},
        "test": {"predict_mean": metrics(y, p_mean), "hedonic_ols": metrics(y, p_ols),
                 "random_forest": metrics(y, p_rf)},
        "train": {"hedonic_ols": metrics(train.y, ols.predict(train)),
                  "random_forest": metrics(train.y, rf.predict(X_train))},
        "forest_oob_R2": float(rf.oob_score_),
    }
    res["R2_gain"] = res["test"]["random_forest"]["R2"] - res["test"]["hedonic_ols"]["R2"]
    res["R2_gain_95ci_bootstrap"] = bootstrap_r2_gain(y, p_ols, p_rf)
    top = y >= np.quantile(y, 0.99)
    res["top1pct_mean_log"] = {"actual": float(y[top].mean()), "hedonic_ols": float(p_ols[top].mean()),
                               "random_forest": float(p_rf[top].mean())}
    res["by_auction_house_R2"] = {
        h: {"n": int((test.source == h).sum()),
            "hedonic_ols": metrics(y[test.source == h], p_ols[test.source == h])["R2"],
            "random_forest": metrics(y[test.source == h], p_rf[test.source == h])["R2"]}
        for h in sorted(test.source.unique())}
    (RESULTS / "rf_vs_ols_metrics.json").write_text(json.dumps(res, indent=2))

    # Importance: which characteristics each model relies on
    X_test = forest_inputs(test, levels)
    rf_groups = {"fame": ["fame"], "area": ["ln_area"],
                 **{c: [k for k in X_test.columns if k.startswith(c + "=")] for c in BASE}}
    ols_groups = {"fame": ["fame"], "area": ["ln_area", "ln_area2"], **{c: [c] for c in BASE}}
    imp = pd.DataFrame({
        "random_forest": permutation_importance(rf.predict, X_test, rf_groups, y),
        "hedonic_ols": permutation_importance(lambda d: ols.predict(d).values, test, ols_groups, y),
    })
    imp.to_csv(RESULTS / "rf_importance.csv")

    # Partial dependence: forest shape vs the OLS functional form
    sample = X_train.sample(3000, random_state=SEED)
    b = ols.params
    ga = np.quantile(train.ln_area, np.linspace(0.02, 0.98, 25))
    pd_area = partial_dependence(rf, sample, "ln_area", ga)
    ols_area = b["ln_area"] * ga + b["ln_area2"] * ga ** 2
    gf = np.unique(np.quantile(train.fame, np.linspace(0, 1, 21)))
    pd_fame = partial_dependence(rf, sample, "fame", gf)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(np.exp(ga), pd_area - pd_area.mean(), color="#B4531A", lw=2.5, label="Random forest")
    ax[0].plot(np.exp(ga), ols_area - ols_area.mean(), "--", color="#3C4043", lw=2, label="Hedonic OLS")
    ax[0].set(xscale="log", xlabel="Area (cm², log scale)", ylabel="Effect on ln(P/A), centred",
              title="Size")
    ax[1].plot(np.expm1(gf) + 1, pd_fame - pd_fame[0], color="#B4531A", lw=2.5, label="Random forest")
    ax[1].plot(np.expm1(gf) + 1, b["fame"] * (gf - gf[0]), "--", color="#3C4043", lw=2, label="Hedonic OLS")
    ax[1].set(xscale="log", xlabel="1 + Wikipedia sitelinks (log scale)",
              ylabel="Effect on ln(P/A) vs. fame = 0", title="Fame")
    for a in ax:
        a.legend(frameon=False)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "rf_partial_dependence.png", dpi=150)

    t = res["test"]
    print("\nAccuracy on the 2026 test set:")
    print(pd.DataFrame(t).T.round(3).to_string())
    lo, hi = res["R2_gain_95ci_bootstrap"]
    print(f"\nR^2 gain of the forest: {res['R2_gain']:.3f} (95% bootstrap interval {lo:.3f} to {hi:.3f})")
    print(f"Training R^2: OLS {res['train']['hedonic_ols']['R2']:.3f}, "
          f"forest {res['train']['random_forest']['R2']:.3f}")
    print("\nPermutation importance (rise in test MSE):")
    print(imp.round(3).to_string())
    print(f"\nwrote results to {RESULTS}")
