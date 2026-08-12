# Input Data Metadata

This page describes the format, structure, and constraints of the data files required to run deltab, whether via the command line or the Python API.

---

## Overview

deltab requires two mandatory inputs and one optional input:

| Input | CLI argument | Python API | Required |
|-------|-------------|------------|----------|
| True chronological ages | `--train-ages` / `--predict-ages` | `ages` parameter | Yes |
| Brain feature matrix (IDPs) | `--train-features` / `--predict-features` | `features` parameter | Yes |
| Confound matrix | *(Python API only)* | `conf` parameter to `train()` / `predict()` | No |

---

## Ages File

### Format

A plain text file containing one chronological age per subject, one value per line. The file is loaded with `numpy.loadtxt` and must resolve to a 1-dimensional array after loading.

| Property | Requirement |
|----------|-------------|
| Shape | (N,) — one value per subject |
| Data type | Float (integer values are accepted and cast) |
| Units | Years |
| Delimiter | Space, comma, or tab — detected automatically |
| Header row | Optional — one header row is skipped automatically if present |
| Missing values | NaN is accepted; subjects with NaN age are removed (see [NaN handling](#nan-handling)) |

### Example

```
72.4
68.1
55.9
61.3
```

Or equivalently with a header:

```
age
72.4
68.1
55.9
61.3
```

### Constraints

- All ages must be in the same unit (years).
- Ages do not need to be sorted.
- There is no hard minimum or maximum age, but the model assumes a roughly continuous age range. Mixing paediatric and elderly subjects in a single training run is not recommended.
- For federated deployments, minimum N ≥ 100 is recommended for meaningful bias correction (see [Disclosure Thresholds](federated/thresholds.md)).

---

## Feature Matrix File

### Format

A plain text file containing one row per subject and one column per imaging-derived phenotype (IDP) or other brain measurement. The file is loaded with `numpy.loadtxt` and must resolve to a 2-dimensional array.

| Property | Requirement |
|----------|-------------|
| Shape | (N, F) — N subjects × F features |
| Data type | Float |
| Units | Any — features are z-score normalised internally before use |
| Delimiter | Space, comma, or tab — detected automatically |
| Header row | Optional — one header row is skipped automatically if present |
| Missing values | NaN accepted per cell; handled by removal or median imputation (see [NaN handling](#nan-handling)) |

### Example (3 subjects, 4 features)

```
0.42  1.13  -0.87  2.01
0.39  1.05  -0.91  1.98
0.51  1.22  -0.79  2.15
```

### Constraints

- The number of rows must exactly match the number of subjects in the ages file.
- The same set of features (same columns, same order) must be used for both training and prediction. deltab will raise an error if `features.shape[1]` differs between `train()` and `predict()`.
- Features do not need to be pre-normalised — deltab computes per-feature mean and standard deviation from the training set and applies the same normalisation to prediction data.
- There is no built-in upper limit on F, but PCA reduction is strongly recommended for F > 50 (see [PCA selection](#pca-component-selection) below).

### Typical IDP sources

| Source | Typical F | Notes |
|--------|-----------|-------|
| FSL BRAID T1 structural pipeline | ~100–300 | Cortical thickness, subcortical volumes, white matter metrics |
| FSL TBSS (diffusion) | ~50–100 | FA, MD, MO per tract |
| FLAIR WMH pipeline | ~10–20 | WMH volume, lesion count per region |
| Combined BRAID IDPs | ~1,000 | Full multi-modal IDP set |

---

## Confound Matrix File

*(Python API only — not available via CLI)*

An optional matrix of confounding variables whose effect should be removed from the feature matrix before model fitting.

| Property | Requirement |
|----------|-------------|
| Shape | (N, C) — N subjects × C confounds |
| Data type | Float |
| Typical confounds | Scanner site, head motion (mean FD), sex (encoded as 0/1) |

Confound regression is performed by projecting the normalised feature matrix onto the confound space and subtracting the fitted component before PCA. The same confound matrix must be passed to `predict()` if it was used during `train()`.

```python
b.train(ages, features, conf=confounds)
delta = b.predict(ages, features, conf=confounds, model=Model.UNBIASED_QUADRATIC)
```

---

## NaN Handling

deltab provides two strategies for subjects with NaN values in the feature matrix, controlled by `--feature-nans` (CLI) or by pre-processing before calling `train()` (Python API):

| Strategy | CLI value | Behaviour |
|----------|-----------|-----------|
| Median imputation *(default)* | `median` | Replaces each NaN with the median value of that feature across all subjects. Subject is retained. |
| Subject removal | `remove` | Removes any subject with at least one NaN feature (or NaN age) from both the ages and features arrays before training. |

!!! warning "NaN removal changes N"
    When using `--feature-nans remove`, the number of subjects passed to the model may be less than the number of rows in your input files. Use `--true-ages-output` to save the ages of the subjects actually used, so downstream analyses remain aligned.

---

## PCA Component Selection

The number of PCA components retained from the feature matrix (J) is controlled by one of four mutually exclusive options:

| Option | CLI | Python | Description |
|--------|-----|--------|-------------|
| Proportion of features | `--feature-proportion 0.1` | `ev_proportion=0.1` | Retain J = round(0.1 × F) components |
| Fixed number | `--feature-num 100` | `ev_num=100` | Retain exactly J components |
| Variance explained | `--feature-var 0.9` | `ev_var=0.9` | Retain components explaining ≥ 90% of variance |
| Kaiser-Guttmann | `--kaiser-guttmann` | `ev_kg=True` | Retain components with eigenvalue > mean eigenvalue |
| None (use all features) | *(omit all)* | *(omit all)* | No PCA — use raw normalised features |

Smith et al. (2019) recommend `ev_proportion=0.1` (10% of features) as a robust default for most neuroimaging datasets. Using all features without PCA is only appropriate when F is small (< 20) or features are already known to be low-dimensional.

---

## Subject Alignment

The ages file and features file must be **row-aligned**: row *i* of the features matrix corresponds to the subject whose age is at position *i* in the ages vector. deltab does not perform any subject matching by ID — alignment is purely positional.

If subjects are removed due to NaN (with `--feature-nans remove`), deltab removes the same indices from both arrays simultaneously, preserving alignment. Use `--true-ages-output` to retrieve the final aligned age vector after removal.

---

## References

- Smith, S.M. et al. (2019). Estimation of brain age delta from brain imaging. *NeuroImage*, 200, 528–539. [https://doi.org/10.1016/j.neuroimage.2019.06.017](https://doi.org/10.1016/j.neuroimage.2019.06.017)
