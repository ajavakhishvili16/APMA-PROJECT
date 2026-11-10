# Random Forest vs. the Hedonic Model

This folder holds the random forest part of our APMA 4903 project. It asks one question:

> Given the same artwork characteristics, does a random forest predict 2026 auction prices
> more accurately than our hedonic OLS model?

Both models are fitted on **21,783 sales from Nov 2023 – Dec 2025** and tested on
**8,458 sales from Jan – Oct 2026** that neither model sees during fitting or tuning.

---

## The fastest way to run it: Google Colab (no installation)

1. Go to **https://colab.research.google.com** and sign in with a Google account.
2. Choose **File → Open notebook → GitHub**, paste
   `https://github.com/ajavakhishvili16/apma-project`, and open
   **`random_forest/rf_vs_ols_colab.ipynb`**.
   (Or download the notebook from this folder and use **File → Upload notebook**.)
3. Choose **Runtime → Run all**.

That's it. The first cell downloads this repository, including the data, so there is nothing
to upload. The whole notebook takes about **5 minutes** and shows every table and chart below
the cells.

What each section of the notebook does:

| Section | What it shows |
|---|---|
| 1 | Downloads the repository and checks that both data files are present |
| 2 | Fits the hedonic OLS model (25 coefficients, standard errors clustered by artist) |
| 3 | Tunes the random forest on training data, fits it, and compares it with OLS on 2026 sales |
| 4 | Results table: R², error measures, and the bootstrap interval for the forest's gain |
| 5 | What the forest learned: size and fame, forest vs. OLS |
| 6 | Validation charts: how the leaf size, inputs per split (m) and number of trees (B) were chosen |
| 7 | Accuracy on 2026 sales as trees are added |

---

## Running it on your own computer

You need **Python 3.10 or newer** (check with `python3 --version`; on Windows use `python`).

1. **Download the repository.** On the GitHub page click **Code → Download ZIP** and unzip it,
   or run `git clone https://github.com/ajavakhishvili16/apma-project.git`.
2. **Open a terminal in this folder** (`apma-project/random_forest`):
   ```bash
   cd path/to/apma-project/random_forest
   ```
3. **Install the libraries** (one time only):
   ```bash
   python3 -m pip install -r requirements.txt
   ```
4. **Run everything:**
   ```bash
   bash run_all.sh
   ```
   On Windows, run the three scripts one by one instead:
   ```bash
   python src/hedonic_ols.py
   python src/random_forest.py
   python src/validation.py
   ```

It takes about **2 minutes**. The tables print in the terminal, and every output is saved in
`results/`.

---

## Files you need

The code reads two data files from the **repository root** (one folder up). Both are
already in this repository; do not rename or move them.

| File (repository root) | Contents |
|---|---|
| `training_set_2023_2025_clean-2.csv` | 21,783 training sales, Nov 2023 – Dec 2025 |
| `test_set_2026_clean-2.csv` | 8,458 test sales, Jan – Oct 2026 |

Everything else is in this folder:

| File | What it does |
|---|---|
| `rf_vs_ols_colab.ipynb` | The Colab notebook: runs all the code below and shows the results |
| `src/common.py` | Shared settings: file locations, the model formula, accuracy measures |
| `src/hedonic_ols.py` | Fits the hedonic OLS model. Reproduces the team's `coefficients-1.csv` exactly |
| `src/random_forest.py` | Tunes and fits the random forest, then compares it with OLS on 2026 sales |
| `src/validation.py` | Draws the validation charts for the leaf size, m and B |
| `run_all.sh` | Runs the three scripts in order |
| `requirements.txt` | The Python libraries needed |
| `results/` | Outputs from our own run, so you can read them without running anything |
| `simulator/forest_simulator.html` | Interactive demonstration: open it in any web browser |
| `simulator/rfcore.js` | The same random forest algorithm in JavaScript, used by the simulator |

---

## The model

**Target:** y = ln(P/A), the log of price per cm² in September-2026 US dollars.

**Characteristics (same for both models):** artist exposure, fame
(ln(1 + Wikipedia sitelinks)), area (ln A and (ln A)²), medium, support, sale season and
creation period.

- **Hedonic OLS:** the shape is chosen by us (linear in fame, quadratic in ln A, additive dummies).
  25 coefficients.
- **Random forest:** the shape is learned from the data. 500 trees; the inputs are fame, ln A and
  one 0/1 indicator per category level, 28 inputs in total.

### How the forest works

```
for b = 1 to B:
    draw n sales with replacement                        # Part 1: bootstrapping
    grow tree T_b from a root node holding the sample:   # Part 3: tree construction
        for each node that can still be split:
            draw m of the p inputs at random              # Part 2: feature selection
            for each drawn input j and cutoff s:
                gain(j, s) = (n_L * n_R / n) * (mean_L - mean_R)^2
            split on the (j, s) with the largest gain, if both sides keep >= l_min sales
    each leaf predicts the mean y of its sales
Predict a new sale x:  y_hat(x) = (1/B) * sum_b T_b(x)    # Part 4: aggregation
                       P_hat = A * exp(y_hat(x))
```

The split gain is the drop in squared error from splitting a node (a regression forest, so
squared error rather than Gini impurity).

### How the settings were chosen (validation)

Only the training sales are used: the forest is fitted on sales **before July 2025** and scored on
sales from **July – December 2025**. The 2026 test set is never used for tuning.

Validation R² for each combination tried (B = 300 trees):

| Min sales per leaf | m = 9 of 28 | m = 14 of 28 | m = 28 of 28 |
|---|---|---|---|
| 1 | 0.373 | 0.380 | 0.374 |
| **5** | **0.426** | 0.423 | 0.409 |
| 10 | 0.425 | 0.424 | 0.415 |
| 20 | 0.416 | 0.417 | 0.411 |
| 40 | 0.402 | 0.407 | 0.398 |

- **Chosen: m = 9, minimum 5 sales per leaf**, the highest validation R². A leaf size of 10 is
  almost identical (0.425), so the forest is not sensitive to this choice.
- **B = 500 trees:** validation R² rises quickly and levels off (0.400 with 10 trees, 0.425 with
  250, 0.427 with 500), so more trees add only computing time.
- scikit-learn turns `max_features=0.33` into int(0.33 × 28) = 9 inputs per split.

`results/validation_curves.png` shows these results as charts.

---

## Results on the 2026 test set

| Model | R² | Median error factor |
|---|---|---|
| Baseline: predict the training mean | ≈ 0 | 3.34× |
| Hedonic OLS (25 coefficients) | 0.380 | 2.48× |
| **Random forest** (B = 500, m = 9, min leaf 5) | **0.438** | **2.40×** |

- **R²** is the share of variation in ln(P/A) explained, compared with predicting the average.
- **Median error factor** is how far off a typical prediction is: 2.4× means the predicted price is
  typically within a factor of about 2.4 of the actual price, in either direction.
- The forest's gain is **0.057 R²**, with a 95% bootstrap interval of **[0.047, 0.067]**. Real, but modest.
- The forest **overfits** somewhat (training R² 0.619 against 0.438 on 2026). OLS does not
  (0.377 against 0.380).
- Most of the forest's advantage is at **Phillips** (R² 0.415 against 0.299 for OLS). At Bonhams
  the gap is smaller (0.429 against 0.392).

Full numbers: `results/rf_vs_ols_metrics.json`.

---

## The interactive simulator

Download `simulator/forest_simulator.html` and open it in any web browser (no internet or
installation needed). It grows a forest tree by tree and shows the four parts of the algorithm:
the bootstrap draw, the split search with every candidate's gain, one artwork's path down a tree,
and R² rising as trees are added.

To run in a browser, it uses a random 3,000 of the training sales and six characteristics
(it leaves out creation period). Its numbers are therefore close to, but not the same as, the
results above.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: ...training_set_2023_2025_clean-2.csv` | Run the code from inside `random_forest/`, and keep both CSVs in the repository root (one folder up) |
| `ModuleNotFoundError: No module named 'sklearn'` (or `statsmodels`, ...) | Run `python3 -m pip install -r requirements.txt` |
| `python3: command not found` (Windows) | Use `python` instead of `python3` |
| `bash: command not found` (Windows) | Run the three `python src/...` commands one by one |
| Numbers differ slightly in the third decimal | Different library versions can change the random draws. Our run used scikit-learn 1.9 |

---

## References

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32.
https://doi.org/10.1023/A:1010933404324

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The elements of statistical learning: Data
mining, inference, and prediction* (2nd ed.). Springer. https://doi.org/10.1007/978-0-387-84858-7

Potrawa, T., & Tetereva, A. (2022). How much is the view from the window worth? Machine
learning-driven hedonic pricing model of the real estate market. *Journal of Business Research,
144*, 50–65. https://doi.org/10.1016/j.jbusres.2022.01.027
