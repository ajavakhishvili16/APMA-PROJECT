# Which House Sold It?

A logistic classifier that predicts whether an artwork was sold by **Bonhams** or
**Phillips**, fitted by **Newton's method** written from scratch in NumPy.

Built as part of a hedonic pricing study of art auction data. The hedonic model asks what
explains the price of an artwork; this turns the question around and asks what the artwork
says about the house. If the houses are highly separable, a "house premium" in a hedonic
regression is partly measuring *what* is sold rather than *where*.

## Results

Trained on 21,783 lots sold 2023–2025, evaluated on 8,458 lots sold in 2026.

| model | features | test AUC | accuracy |
|---|---|---|---|
| A — artwork characteristics only | 29 | 0.858 | 85.0% |
| B — + pre-sale estimates | 32 | **0.891** | 87.0% |
| C — + realized price | 34 | 0.890 | 87.0% |

Baseline (always guess Bonhams): 78.5%.

Three findings:

1. **The houses are separable from the art alone.** No price information is needed to reach
   AUC 0.86 out of sample.
2. **The split is chronological.** Work created after 2000 has roughly nine times the odds of
   being Phillips; pre-1800 work is 100% Bonhams in the training data.
3. **The realized price adds nothing once estimates are known** (−0.0005 AUC). The pre-sale
   estimate already encodes the house's valuation.

## Newton's method

Logistic regression has no closed-form solution, unlike the hedonic OLS it sits beside:

| | hedonic OLS | this classifier |
|---|---|---|
| model | ln(price) = x'β + ε | P(Phillips) = σ(x'β) |
| solution | closed form, β = (X'X)⁻¹X'y | none — the sigmoid makes it nonlinear |

Newton's method uses the curvature as well as the slope:

```
g = X'(y − p)                 gradient
H = X'WX + λI                 Hessian, W = diag(p(1−p))
β ← β + H⁻¹g                  update
```

Each step is a weighted least-squares problem whose weights change every iteration, which is
why the method is also called **IRLS**. It converges in **12 iterations**; gradient descent is
still short of the optimum after 3,000.

The implementation never forms `H⁻¹` — it solves `H·Δ = g`, which is faster and numerically
steadier.

### Why the ridge term is load-bearing

All 348 lots created before 1800 were sold by Bonhams, and none by Phillips. No finite
coefficient fits that, so the unpenalised maximum-likelihood estimate is −∞. Without `λI`,
Newton's method marches that coefficient toward infinity and the Hessian becomes
ill-conditioned:

| λ | iterations | β(pre-1800) |
|---|---|---|
| 1 | 12 | −0.47 |
| 1e−4 | 21 | −1.46 |
| 1e−8 | 120 (capped) | −2.54 |

## Files

| file | what it is |
|---|---|
| `auction_house_classifier.ipynb` | the full analysis — run this first |
| `build_lots.py` | fits the model and exports `lots.json` for the animation |
| `index.html` | animated walkthrough of 18 real 2026 lots, self-contained |
| `auction_train_with_estimates.csv` | 21,783 lots, Nov 2023 – Dec 2025 |
| `auction_test_with_estimates.csv` | 8,458 lots, Jan – Oct 2026 |

## Running it

Open the notebook in [Google Colab](https://colab.research.google.com) (File → Upload
notebook), run the first cell, upload the two CSVs when prompted, then Runtime → Run all.

Locally:

```bash
pip install numpy pandas scikit-learn matplotlib
python build_lots.py
```

Open `index.html` in any browser — no server needed.

## Data notes

The cleaned train/test split is time-based, so no sale straddles the boundary. Pre-sale
estimates were merged back in from the source data on `source_url`, with 99% coverage.

Thirteen columns were dropped as leakage before modelling. They identify the house by
scraping artefacts rather than by anything about the art:

- `source`, `source_url` — the label itself
- `currency` — AUD and EUR lots are 100% Bonhams
- `price_realized`, `fx_to_usd` — reconstruct the currency
- `cpi_sale_month`, `inflation_factor` — deterministic functions of the sale month
- `ln_ppa` — exactly `ln(price) − ln_area`, which makes the Hessian singular
- `sitelinks` — exactly `exp(fame) − 1`, and 40% missing
- `cluster` — 8,372 artist levels, 34% of test artists unseen in training
- `artist_name`, `artwork_title`, `medium` — free text whose formatting differs by house

The implementation is verified against scikit-learn's own solver on the same penalised
objective: coefficients agree to 6.5e−11.

## Note on the animation

The lot images are not reproduced. Each work is drawn to scale from its catalogued dimensions
against a shared 1-metre reference, and links out to its listing on the auction house's own
site. The eighteen lots span the probability range and are not a random sample, so the
on-screen tally is illustrative; the figures in the table above are the real ones.
