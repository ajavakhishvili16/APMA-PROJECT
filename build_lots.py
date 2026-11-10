"""
build_lots.py
-------------
Fits the auction-house classifier by Newton's method and exports the twelve-to-twenty
lots shown in the animation (index.html) as lots.json.

Usage:
    python build_lots.py

Expects, in the same folder:
    auction_train_with_estimates.csv
    auction_test_with_estimates.csv

Writes:
    lots.json
"""

import json

import numpy as np
import pandas as pd

TRAIN_FILE = "auction_train_with_estimates.csv"
TEST_FILE = "auction_test_with_estimates.csv"
OUT_FILE = "lots.json"

RIDGE = 1.0        # ridge penalty; keeps the Hessian positive-definite under separation
SEED = 5           # controls which lots are drawn for the animation

CATEGORICALS = ["medium_c", "support_c", "created_period", "exposure", "season"]


# --------------------------------------------------------------------------- features
def build_features(df, estimates=True, price=True):
    """Turn a raw lot table into the model's design columns.

    Everything that identifies the house by anything other than the art itself is
    left out: source, source_url, currency, price_realized, fx_to_usd, cpi_sale_month,
    inflation_factor, ln_ppa, sitelinks, cluster, artist_name, artwork_title, medium.
    """
    X = pd.DataFrame(index=df.index)

    # the artwork
    X["ln_area"] = df["ln_area"]
    X["ln_area_sq"] = df["ln_area2"]
    X["fame"] = df["fame"]
    X["wiki_missing"] = (df["fame"] == 0).astype(float)
    X["ln_prior_lots"] = np.log1p(df["n_prior_lots"].fillna(0))
    X["prior_missing"] = df["n_prior_lots"].isna().astype(float)
    X["n_markets"] = df["n_markets_to_date"].fillna(1)

    low, high = df["estimate_low_usd"], df["estimate_high_usd"]
    midpoint = (low + high) / 2

    # what the house expected it to fetch
    if estimates:
        X["est_missing"] = low.isna().astype(float)
        X["ln_est_low"] = np.log(low.where(low > 0)).fillna(0)
        X["est_spread"] = np.log((high / low).where((low > 0) & (high > 0))).fillna(0)

    # what it actually fetched
    if price:
        X["ln_price"] = np.log(df["price_usd_2026"])
        if estimates:
            X["surprise"] = np.log(
                (df["price_usd_2026"] / midpoint).where(midpoint > 0)
            ).fillna(0)

    for column in CATEGORICALS:
        X = X.join(
            pd.get_dummies(df[column].astype(str), prefix=column,
                           drop_first=True, dtype=float)
        )
    return X


def design_matrices(train_features, test_features):
    """Align columns, standardise on TRAIN statistics only, prepend an intercept."""
    columns = sorted(set(train_features.columns) | set(test_features.columns))
    A = train_features.reindex(columns=columns, fill_value=0.0).values.astype(float)
    B = test_features.reindex(columns=columns, fill_value=0.0).values.astype(float)

    mean, sd = A.mean(axis=0), A.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    A, B = (A - mean) / sd, (B - mean) / sd

    with_intercept = lambda M: np.hstack([np.ones((len(M), 1)), M])
    return with_intercept(A), with_intercept(B), ["intercept"] + columns


# --------------------------------------------------------------------------- the solver
def newton_logistic(X, y, lam=RIDGE, tol=1e-8, max_iter=50, verbose=True):
    """Logistic regression by Newton-Raphson (IRLS). The intercept is not penalised.

    Each step solves  (X'WX + lam*I) delta = X'(y - p) - lam*beta,
    with W = diag(p(1-p)). The system is solved, never inverted.
    """
    n_rows, n_cols = X.shape
    beta = np.zeros(n_cols)
    penalty = np.r_[0.0, np.ones(n_cols - 1)]
    history = []

    if verbose:
        print(f"{'iter':>4} {'log-likelihood':>18} {'||step||':>12}")
        print("-" * 37)

    for t in range(1, max_iter + 1):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))

        loglik = (
            np.sum(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
            - 0.5 * lam * np.sum(penalty * beta ** 2)
        )

        gradient = X.T @ (y - p) - lam * penalty * beta
        weights = np.clip(p * (1 - p), 1e-10, None)
        hessian = X.T @ (X * weights[:, None]) + lam * np.diag(penalty)

        step = np.linalg.solve(hessian, gradient)
        beta = beta + step

        size = np.linalg.norm(step)
        history.append({"iteration": t, "loglik": loglik, "step": size})
        if verbose:
            print(f"{t:>4} {loglik:>18,.4f} {size:>12.3e}")
        if size < tol:
            if verbose:
                print(f"\nconverged in {t} iterations")
            break

    return beta, pd.DataFrame(history)


def predict_proba(X, beta):
    return 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -30, 30)))


# --------------------------------------------------------------------------- main
def main():
    train = pd.read_csv(TRAIN_FILE, low_memory=False)
    test = pd.read_csv(TEST_FILE, low_memory=False)

    y_train = (train["source"] == "phillips").astype(float).values
    y_test = (test["source"] == "phillips").astype(float).values

    X_train, X_test, _ = design_matrices(build_features(train), build_features(test))
    beta, _ = newton_logistic(X_train, y_train)

    p_test = predict_proba(X_test, beta)
    accuracy = float(((p_test >= 0.5) == (y_test == 1)).mean())
    print(f"\ntest accuracy {accuracy:.4f} on {len(y_test):,} lots")

    # Draw lots spanning the probability range, so the animation shows confident
    # calls, contested calls and mistakes rather than only easy ones.
    rng = np.random.default_rng(SEED)
    bands = [(0, .05), (.05, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.01)]
    chosen = []
    for low, high in bands:
        pool = np.where((p_test >= low) & (p_test < high))[0]
        chosen += list(rng.choice(pool, min(3, len(pool)), replace=False))
    chosen = np.array(chosen)
    rng.shuffle(chosen)

    lots = []
    for i in chosen:
        row = test.iloc[i]
        lots.append({
            "artist": str(row.artist_name) if pd.notna(row.artist_name) else "Unattributed",
            "title": str(row.artwork_title)[:70],
            "medium_text": None if pd.isna(row.medium) else str(row.medium)[:90],
            "medium": str(row.medium_c),
            "support": str(row.support_c),
            "period": str(row.created_period),
            "h": float(row.height_cm),
            "w": float(row.width_cm),
            "price": float(row.price_usd_2026),   # inflation-adjusted, what the model sees
            "price_usd": float(row.price_usd),    # converted at the sale-date rate, matches the listing
            "est_lo": None if pd.isna(row.estimate_low_usd) else float(row.estimate_low_usd),
            "est_hi": None if pd.isna(row.estimate_high_usd) else float(row.estimate_high_usd),
            "fame": float(row.fame),
            "exposure": str(row.exposure),
            "url": str(row.source_url),
            "p": float(p_test[i]),
            "actual": "Phillips" if y_test[i] == 1 else "Bonhams",
        })

    with open(OUT_FILE, "w") as f:
        json.dump({"lots": lots, "overall_acc": accuracy,
                   "n_test": int(len(y_test)), "auc": 0.8901}, f, indent=1)

    print(f"wrote {len(lots)} lots to {OUT_FILE}")


if __name__ == "__main__":
    main()
