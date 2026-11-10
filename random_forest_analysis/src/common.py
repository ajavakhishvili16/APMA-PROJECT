"""Shared settings and helpers for the analysis scripts."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # .../apma-project/random_forest
DATA = ROOT.parent                                  # the repository root, where the CSVs live
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

TRAIN = DATA / "training_set_2023_2025_clean-2.csv"   # 21,783 sales, Nov 2023 - Dec 2025
TEST = DATA / "test_set_2026_clean-2.csv"             # 8,458 sales, Jan - Oct 2026

# Categorical characteristics and their base (omitted) levels
BASE = {"exposure": "1-lot", "medium_c": "oil", "support_c": "canvas", "season": "Q1",
        "created_period": "1945_69"}

# Hedonic model (team's final specification): ln(P/A) on the five categorical
# characteristics above, fame, ln A and (ln A)^2 -- 25 coefficients.
FORMULA = ("ln_ppa ~ C(exposure, Treatment('1-lot')) + C(medium_c, Treatment('oil'))"
           " + C(support_c, Treatment('canvas')) + C(season, Treatment('Q1'))"
           " + C(created_period, Treatment('1945_69'))"
           " + fame + ln_area + ln_area2")

SEED = 0


def load(path):
    """Read a data file and add the target y = ln(price per cm^2, 2026 USD)."""
    d = pd.read_csv(path, low_memory=False)
    d["y"] = np.log(d.price_usd_2026 / d.area_cm2)
    d["date"] = pd.to_datetime(d.auction_date)
    return d


def metrics(y, pred):
    """Out-of-sample accuracy on the log scale, plus two price-scale summaries."""
    y, pred = np.asarray(y), np.asarray(pred)
    e = y - pred
    return {
        "R2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()),
        "RMSE_log": float(np.sqrt((e ** 2).mean())),
        "MAE_log": float(np.abs(e).mean()),
        # median factor between predicted and actual price (2.0 = off by 2x)
        "typical_miss_factor": float(np.exp(np.median(np.abs(e)))),
        "share_within_1.5x": float(np.mean(np.abs(e) < np.log(1.5))),
    }
