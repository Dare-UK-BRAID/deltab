# Examples

The `examples/` directory contains Python scripts that reproduce the simulation results from Smith et al. (2019). They serve both as validation of the deltab implementation and as practical demonstrations of how to use the Python API.

All example scripts share utility functions from `examples/utils.py`.

---

## Running the examples

From the `examples/` directory:

```bash
cd examples
python smith_results_fig2.py    # Figure 2 simulation
python smith_results_fig4.py    # Figure 4 quadratic simulation
python smith_results_table1.py  # Table 1 cross-validation (linear)
python smith_results_table2.py  # Table 2 cross-validation (quadratic)
```

Plots are displayed interactively. Table scripts print results to stdout and write a `table1.csv` file. Pass `--plot` to table scripts to also render the table as a figure.

---

## Figure 1 — Regression Bias Illustration (`smith_results_fig1.py`)

This script does **not** use the `deltab` package. It is a standalone illustration of how ordinary least squares regression introduces bias under several common conditions.

Six scatter plots are generated, each showing a simple case where regressing variable B onto A (or A onto B) produces a biased result:

| Panel | Scenario | Bias mechanism |
|-------|----------|---------------|
| A | Standard Gaussian A and B with noise | Baseline |
| B | Extra noise added to B | Regression dilution on the predicted variable |
| C | Truncation of A (select only middle values) | Cohort selection / range restriction |
| D | Regularised fit (demeaned A and B) | Regression to the mean after demeaning |
| E | Noise added to A | Regression dilution on the predictor |
| F | Truncation of B | Selection on the predicted variable |

Each panel plots a density-coloured scatter of x vs y and overlays the identity line. Departures from the identity line reveal the bias. This corresponds directly to Figure 1 in Smith et al. (2019) and provides the conceptual motivation for the corrected models.

---

## Figure 2 — Simulation 1: Linear Ageing (`smith_results_fig2.py`)

This script reproduces Figure 2 of Smith et al., demonstrating the bias of the simple model and the effectiveness of the unbiased correction in a linear simulation.

### Simulation setup

```python
N = 20000                          # subjects
Y  = 60 + 25*(rand - 0.5) + noise  # age ~ Uniform(45-75) with small noise
DELTA_TRUE = 2 * normal(N)         # true delta ~ N(0, 2 years)
BRAIN_AGE_TRUE = Y + DELTA_TRUE    # brain age = true age + true delta

# 5 underlying features, first is brain age, rest are random
# Mixed via sparse random matrix → 100 observed imaging features
# Measurement noise σ = 0.2 added
```

The simulation constructs a realistic feature matrix where brain age is embedded in one underlying component, mixed with random noise through a sparse matrix. This mimics the structure of real IDP datasets.

### What is shown

Six scatter plots arranged in a 2×3 grid compare the simple model (top row) and unbiased model (bottom row):

| Column | x-axis | y-axis | What it shows |
|--------|--------|--------|--------------|
| 1 | True age Y | Predicted age Y_B1 | How well age is recovered |
| 2 | True age Y | Delta δ | The age-dependence bias |
| 3 | True delta | Predicted delta | Whether the biological signal is recovered |

In the top row (simple model), column 2 shows a strong negative correlation between true age and predicted delta — the classic bias. Younger subjects appear to have positive delta, older subjects negative delta, regardless of their true brain age. In the bottom row (unbiased model), this correlation is eliminated.

### Key code

```python
from deltab import BrainDelta, Model

b = BrainDelta()
b.train(Y, X)

# Simple model — biased
y1 = b.predict(Y, X, model=Model.SIMPLE, return_delta=False)
d1 = b.predict(Y, X, model=Model.SIMPLE, return_delta=True)

# Unbiased model — corrected
y2 = b.predict(Y, X, model=Model.UNBIASED, return_delta=False)
d2 = b.predict(Y, X, model=Model.UNBIASED, return_delta=True)
```

---

## Figure 4 — Simulation 2: Quadratic Ageing (`smith_results_fig4.py`)

This script reproduces Figure 4, which extends the simulation to include a quadratic component in the true brain age trajectory.

### Simulation setup

The quadratic term is constructed to be orthogonal to the linear age effect, so it adds genuinely non-linear curvature:

```python
Yquad = 0.05                             # quadratic coefficient
Y_DEMEAN = demean(Y)
Y2 = demean(Yquad * Y_DEMEAN**2)
# Orthogonalise Y2 with respect to Y_DEMEAN
Y2 = Y2 - Y_DEMEAN * (pinv(Y_DEMEAN) @ Y2)
BRAIN_AGE_TRUE = Y + DELTA_TRUE + Y2    # includes quadratic component
```

### What is shown

The same 2×3 layout as Figure 2, but comparing `Model.SIMPLE` against `Model.UNBIASED_QUADRATIC`. When a true quadratic ageing effect is present, the linear unbiased model leaves residual quadratic bias; the quadratic correction removes it.

---

## Cross-validation

### Table 1 — Linear simulation (`smith_results_table1.py`)

This script runs a k-fold cross-validation simulation and tabulates key performance metrics for all five models across a range of PCA component counts (J values).

### Cross-validation implementation

```python
def DeltaCV(X, Y, J, Ncv=10):
    delta1 = np.zeros_like(Y)   # simple
    delta2 = np.zeros_like(Y)   # unbiased
    delta3 = np.zeros_like(Y)   # alternate
    delta2q = np.zeros_like(Y)  # unbiased quadratic
    delta3q = np.zeros_like(Y)  # alternate quadratic

    SUBJECT_GROUPS = np.random.randint(Ncv, size=len(Y))   # assign to folds

    b = BrainDelta()
    for i in range(Ncv):
        TRAINING_SET  = (SUBJECT_GROUPS != i)
        PREDICTION_SET = (SUBJECT_GROUPS == i)

        b.train(Y[TRAINING_SET], X[TRAINING_SET, :], ev_num=J)

        for each model:
            delta[PREDICTION_SET] = b.predict(Y[PREDICTION_SET],
                                               X[PREDICTION_SET, :],
                                               model=..., return_delta=True)
    return delta1, delta2, delta3, delta2q, delta3q
```

In each fold, the model is trained **only on subjects not in the held-out set**. Predictions are made for the held-out set using the training subjects' normalisation parameters and PCA. This ensures a true out-of-sample evaluation.

### Metrics reported

For each combination of J (number of PCA components) and model, the script reports:

| Metric | Description |
|--------|-------------|
| `mean_delta` | Mean absolute brain age delta across subjects — should be close to the true σ=2 years |
| `corr_delta` | Correlation between predicted delta and true delta — higher is better |
| `corr_age` | Correlation between predicted delta and true age — should be near zero for corrected models |

The simulation is repeated 20 times and results are averaged. This corresponds directly to Table 1 in Smith et al. (2019), showing that the corrected models dramatically outperform the simple model at higher J values.

!!! note "Runtime"
    With `NUM_SIMULATIONS=20`, `NUM_SUBJECTS=20000`, and several J values, this script takes several minutes to run. Reduce these constants for quicker iteration during development.

### Table 2 — Quadratic simulation (`smith_results_table2.py`)

The same cross-validation structure, extended to test three levels of quadratic ageing effect (`Yquad = 0, 0.01, 0.025`, corresponding to 0, 4, and 10 years of total quadratic deviation from linear ageing). Results are tabulated for all five models including the quadratic variants.

---

## Utility functions (`examples/utils.py`)

| Function | Description |
|----------|-------------|
| `dscatter(ax, x, y)` | Density-coloured scatter plot using kernel density estimation |
| `do_plot(ax, x, y, axlim)` | Convenience wrapper for scatter + identity/zero line |
| `do_table(ax, df)` | Render a pandas DataFrame as a matplotlib table |
| `demean(x)` | Subtract column means from an array |
| `normalize(x)` | Demean and scale to unit standard deviation per column |

These utilities mirror the preprocessing steps in `BrainDelta.train()`, allowing the examples to construct synthetic feature matrices with controlled properties.
