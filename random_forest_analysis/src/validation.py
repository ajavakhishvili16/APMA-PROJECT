"""Validation curves for the random forest settings (B, m, l_min).

    python src/validation.py        (run src/random_forest.py first)

Uses only the training set, split by date:
  fit:        sales before 1 July 2025
  validation: sales from July to December 2025
The 2026 test set is not used.

Left panel:  validation R^2 against the minimum leaf size, one line per m
             (read from results/rf_tuning.csv, written by random_forest.py).
Right panel: validation R^2 against the number of trees B, for the chosen
             (m, l_min): one forest of 1,000 trees, averaged over its first B trees.

Writes results/validation_curves.png and results/validation_B_curve.csv.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from common import BASE, RESULTS, SEED, TRAIN, load, metrics
from random_forest import forest_inputs, tune

MAX_TREES = 1000
B_CHOSEN = 500

if __name__ == "__main__":
    train = load(TRAIN)
    levels = {c: sorted(train[c].astype(str).unique()) for c in BASE}
    p = len(forest_inputs(train.head(1), levels).columns)

    tuning_file = RESULTS / "rf_tuning.csv"
    tuning = pd.read_csv(tuning_file) if tuning_file.exists() else tune(train, levels)
    tuning["m"] = [max(1, int(f * p)) for f in tuning.max_features]   # scikit-learn's rounding
    best = tuning.sort_values("valid_R2", ascending=False).iloc[0]
    m_best, leaf_best = int(best.m), int(best.min_samples_leaf)

    # R^2 on the validation sales as trees are added
    fit_part = train[train.date < "2025-07-01"]
    valid = train[train.date >= "2025-07-01"]
    rf = RandomForestRegressor(n_estimators=MAX_TREES, min_samples_leaf=leaf_best,
                               max_features=float(best.max_features), n_jobs=-1, random_state=SEED)
    rf.fit(forest_inputs(fit_part, levels), fit_part.y)
    Xv = forest_inputs(valid, levels).values
    running = np.zeros(len(valid))
    r2_by_B = []
    for b, tree in enumerate(rf.estimators_, start=1):
        running += tree.predict(Xv)
        r2_by_B.append(metrics(valid.y, running / b)["R2"])
    pd.DataFrame({"B": range(1, MAX_TREES + 1), "valid_R2": r2_by_B}).to_csv(
        RESULTS / "validation_B_curve.csv", index=False)

    # Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    styles = {6: ("#B4531A", "-", "o"), 10: ("#3C4043", "--", "s"), 21: ("#8A8A85", ":", "^")}
    for m, g in tuning.groupby("m"):
        g = g.sort_values("min_samples_leaf")
        c, ls, mk = styles.get(m, ("gray", "-", "o"))
        ax1.plot(g.min_samples_leaf, g.valid_R2, ls=ls, marker=mk, color=c, lw=2.2, label=f"m = {m} of {p}")
    ax1.plot(leaf_best, best.valid_R2, "o", ms=14, mfc="none", mec="#B4531A", mew=2.5)
    ax1.annotate(f"chosen: m = {m_best}, leaf = {leaf_best}\nR² = {best.valid_R2:.3f}",
                 (leaf_best, best.valid_R2), xytext=(30, -95), textcoords="offset points",
                 arrowprops=dict(arrowstyle="-", color="#3C4043"), fontsize=10)
    leaves = sorted(tuning.min_samples_leaf.unique())
    ax1.axvspan(min(leaves) * 0.8, 3, color="#F5C4B3", alpha=0.45, lw=0)
    ax1.text(1.55, ax1.get_ylim()[0] + 0.004, "overfits", ha="center", fontsize=10, color="#712B13")
    ax1.set(xscale="log", xlabel="Minimum sales per leaf (ℓ_min)", ylabel="Validation R² (Jul–Dec 2025)",
            title="Choosing m and ℓ_min (B = 300)")
    ax1.set_xticks(leaves)
    ax1.set_xticklabels([int(v) for v in leaves])
    ax1.legend(frameon=False, loc="lower right")

    B = np.arange(1, MAX_TREES + 1)
    ax2.plot(B, r2_by_B, color="#B4531A", lw=2.2)
    ax2.axvspan(250, MAX_TREES, color="#E3A962", alpha=0.2, lw=0)
    ax2.text(480, min(r2_by_B) + 0.01, "flat beyond\n≈ 250 trees", ha="center", fontsize=10)
    ax2.plot(B_CHOSEN, r2_by_B[B_CHOSEN - 1], "o", ms=10, color="#B4531A")
    ax2.annotate(f"B = {B_CHOSEN}: R² = {r2_by_B[B_CHOSEN - 1]:.3f}", (B_CHOSEN, r2_by_B[B_CHOSEN - 1]),
                 xytext=(-40, -40), textcoords="offset points", fontsize=10,
                 arrowprops=dict(arrowstyle="-", color="#3C4043"))
    ax2.set_xscale("log")
    ax2.set_xticks([1, 10, 100, 250, 500, 1000])
    ax2.set_xticklabels(["1", "10", "100", "250", "500", "1000"])
    ax2.set(xlabel="Number of trees averaged (B)", ylabel="Validation R² (Jul–Dec 2025)",
            title=f"Choosing B (m = {m_best}, ℓ_min = {leaf_best})")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS / "validation_curves.png", dpi=150)

    print(tuning.pivot(index="min_samples_leaf", columns="m", values="valid_R2").round(3).to_string())
    print(f"\nchosen: m = {m_best}, min leaf = {leaf_best}, validation R^2 = {best.valid_R2:.4f}")
    for b in [1, 5, 10, 50, 100, 250, 500, 1000]:
        print(f"B = {b:>4}: validation R^2 {r2_by_B[b - 1]:.4f}")
    print(f"\nwrote {RESULTS / 'validation_curves.png'}")
