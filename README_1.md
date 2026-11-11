# Pricing Model for Art Auction Sales

APMA 4903 Seminar, November 2026 · Jana Choe, Anastasia Javakhishvili, Amy Seo, Asher Strell

A hedonic pricing model for paintings and works on paper sold at Bonhams and Phillips
(November 2023 – October 2026). We estimate how artist exposure, fame, size, medium, support,
creation period and sale season affect price per cm², using OLS with standard errors
clustered by artist.

## Model

```math
\ln\left(\frac{P_i}{A_i}\right) = \alpha + \gamma\cdot\text{Exposure}_i + \beta_F\,\text{Fame}_i + \beta_1 \ln A_i + \beta_2 (\ln A_i)^2 + \mu\cdot\text{Medium}_i + \lambda\cdot\text{Support}_i + \theta\cdot\text{CreationPeriod}_i + \delta\cdot\text{SaleSeason}_i + \varepsilon_i
```

| Variable | Coding | Base |
|---|---|---|
| Exposure | 4 tiers: 1-lot, local, circulating, established | 1-lot |
| Fame | ln(1 + number of Wikipedia language editions) | – |
| Size | ln A and (ln A)², A = height × width in cm² | – |
| Medium | oil, acrylic, watercolour, ink, drawing, mixed, other | oil |
| Support | canvas, paper, panel, other | canvas |
| Creation period | pre-1800, 1800s, 1900–44, 1945–69, 1970–99, 2000+, unknown | 1945–69 |
| Sale season | Q1–Q4 | Q1 |

- **Estimator:** OLS, β̂ = (X′X)⁻¹X′y (25 coefficients).
- **Standard errors:** cluster-robust by artist (8,372 clusters),
  V̂ = (X′X)⁻¹ [Σ_g X_g′ ε̂_g ε̂_g′ X_g] (X′X)⁻¹.
- **Joint tests:** Wald, W = (Rβ̂)′(RV̂R′)⁻¹(Rβ̂), one per variable block.
- **% effects:** Kennedy (1981) correction for dummies, 100·(e^(β̂ − ½SE²) − 1), which differs by less than 2 percentage points; due to the small difference, the presentation uses the uncorrected values due to time limitation.

- 

## Data

- 30,241 sold lots from Bonhams and Phillips, converted to USD at the sale-date exchange rate.
- Prices adjusted to September 2026 US dollars using CPI-U (US city average, not seasonally
  adjusted, BLS). October 2025 CPI was not published, so the average of September and
  November 2025 is used.
- **Training set:** sales before 1 Jan 2026 (21,783 sales, Nov 2023 – Dec 2025).
- **Test set:** all 2026 sales (8,458 sales).
- Lots with no artist name (1,584) are coded as exposure = 1-lot, fame = 0, and each forms
  its own cluster.

## Files

| File | What it is |
|---|---|
| `final_hedonic_model_OLS:clusteredSE:waldtest.py` | Main script: train/test split, OLS with clustered SEs, % effects, joint Wald tests, size elasticities |
| `cpi inflation adjustment to september.py` | How prices were adjusted to September 2026 USD (documents the CPI table; its output is `dataset_inflation_adjusted.csv`) |
| `dataset_inflation_adjusted.csv` | Full cleaned dataset with inflation-adjusted prices (input to the main script) |
| `training_set_2023_2025_clean-2.csv` | Training sample (2023–2025) |
| `test_set_2026_clean-2.csv` | Prediction sample (2026) |
| `coefficients-1.csv` | Coefficients, clustered SEs, t, p, 95% CIs, Kennedy % effects |
| `wald_tests-1.csv` | Joint Wald tests per block, with R² and AIC/BIC lost when the block is removed |
| `simulation_auction-room.html` | Same demo as a standalone web page; open in any browser |

## How to run

Requires Python 3.9+.

```bash
pip install numpy pandas statsmodels scipy
python "final_hedonic_model_OLS:clusteredSE:waldtest.py"
```

The script reads `dataset_inflation_adjusted.csv` from the current folder and downloads it
from this repository if it is missing. To use another file, pass its path:

```bash
python "final_hedonic_model_OLS:clusteredSE:waldtest.py" path/to/data.csv
```

It prints the results and writes `coefficients.csv`, `wald_tests.csv` and
`test_set_2026.csv`.

**Auction Room demo:** open `Auction_Room.ipynb` in Google Colab and run all cells, or open
`simulation_auction-room.html` in a browser.

## Key results

R² = 0.377 (adjusted 0.376), n = 21,783.

| Effect (vs base) | % effect | 95% CI |
|---|---|---|
| Local artist | +54% | +41% to +69% |
| Circulating artist | +95% | +72% to +122% |
| Established artist | +129% | +83% to +191% |
| Fame, doubling Wikipedia editions | about +10% | +6% to +13% |
| Drawing (vs oil) | −52% | −59% to −43% |
| Paper (vs canvas) | −46% | −53% to −37% |
| Q2 / Q4 sale (vs Q1) | +46% / +49% | |

Every variable block is jointly significant (Wald p < 0.001). Size matters most: removing
it lowers R² by 0.22. Price per cm² falls with size, so larger works cost more in total but
less per cm².

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
