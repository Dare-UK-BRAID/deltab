# Egress & Privacy Risk Analysis

This page provides a line-by-line audit of the deltab codebase (`deltab/deltab.py`, `deltab/main.py`) for data that risk leaving a TRE node and reaching a central aggregator, in violation of DPUK data governance requirements and TREvolution/SACRO egress principles.

---

## Egress Risk Classification

Each variable or output is classified as follows:

| Class | Meaning |
|-------|---------|
| **HIGH** | Contains raw subject-level data or can directly re-identify subjects. Must never leave the node without explicit DAC approval and SDC review. |
| **MEDIUM** | Contains derived statistics that could enable inference attacks if N is small or cohort is unusual. Requires SDC review before egress. |
| **LOW** | Aggregate parameters unlikely to re-identify subjects in most conditions. Still requires disclosure control audit. |

---

## Variable-Level Audit

### Raw inputs retained on the `BrainDelta` object

After `train()` completes, the following subject-level arrays are stored as instance attributes and are therefore serialised into the pickle file produced by `save()`:

| Variable | Line | Risk | Description |
|----------|------|------|-------------|
| `self.ytrain` | 62 | **HIGH** | Full vector of true chronological ages, one per subject. Ages are quasi-identifying — in small or specialist cohorts they may uniquely identify individuals when combined with other attributes. |
| `self.xtrain` | 68 | **HIGH** | Full raw IDP feature matrix (N × F). Contains all imaging-derived phenotypes for every training subject. Transmission to a central server is equivalent to transmitting the imaging data itself. |
| `self.x_norm` | 81 | **HIGH** | Normalised (zero mean, unit std) feature matrix — still per-subject. Rescaling does not remove re-identification risk. |
| `self.x_reduced` | 113 | **HIGH** | PCA-reduced feature matrix (N × J). Per-subject embeddings in the reduced space. Dimensionality reduction does not prevent inversion when PCA components are also known. |
| `self.y_demean` | 79 | **HIGH** | Demeaned per-subject ages. Reversible to true age given `self.y_mean`. |
| `self.ysq`, `self.ysq_demean`, `self.ysq_orth` | 121–125 | **MEDIUM** | Per-subject quadratic age terms. Derived from `ytrain`; can be inverted to recover approximate ages. |
| `self.y2`, `self.y2sq` | 128–131 | **MEDIUM** | Stacked age matrices used in bias correction. Per-subject. |
| `self.num_subjects_train` | 65 | **MEDIUM** | Exact training N. In small TREs this can be a disclosure risk on its own (see [Thresholds](thresholds.md)). |

### Model parameters — what should be transmitted

These are the **only** variables that should be eligible for egress (after SDC review):

| Variable | Line | Risk | Description |
|----------|------|------|-------------|
| `self.b1` | 135 | **LOW** | Regression coefficients β₁ (shape J,). These map PCA features to age. Do not contain per-subject data directly. Risk increases if J is small or N is small. |
| `self.b2`, `self.b2sq` | 143–144 | **LOW** | Bias correction coefficients (scalars or shape 2,). Aggregate parameters. |
| `self.gamma`, `self.gammasq` | 147–148 | **LOW** | Alternate model coefficients (shape J × F). Similar risk profile to b1. |
| `self.pca.components_` | — | **MEDIUM** | PCA eigenvectors (shape J × F). These are eigenvectors of the training data covariance matrix. In small cohorts they can leak information about the feature distribution, and in combination with `b1` they enable model inversion. |
| `self.x_mean`, `self.x_std` | 81 | **LOW** | Per-feature mean and standard deviation (shape F,). Population-level statistics — low re-identification risk for large N. |
| `self.y_mean` | 78 | **LOW** | Scalar mean of training ages. Low risk for large N. |
| `self.conf_beta` | 86 | **LOW** | Confound regression coefficients. Population-level. |

---

## Critical Egress Risk: `save()` and `save_text()`

### `save()` — pickle serialisation

```python
# deltab/deltab.py line 169
def save(self, fname):
    pickle.dump(self, open(fname, "wb"))
```

`pickle.dump(self, ...)` serialises the **entire** `BrainDelta` object, including all HIGH-risk variables listed above. If this file is transmitted to a central server, it is equivalent to transmitting the full raw training dataset.

**Required control:** The `save()` output must **never** be approved for egress from a TRE node. A separate `save_weights_only()` method (not currently implemented) should be created that serialises only `b1`, `b2`, `b2sq`, `gamma`, `gammasq`, `x_mean`, `x_std`, `y_mean`, and `pca.components_`, explicitly excluding all per-subject arrays.

### `save_text()` — explicit raw data export

```python
# deltab/deltab.py lines 188–196
output_data = [
    ("ytrain", self.ytrain),   # HIGH — raw ages
    ("xtrain", self.xtrain),   # HIGH — raw features
    ("xnorm", self.x_norm),    # HIGH — normalised features
    ("ynorm", self.y_demean),  # HIGH — demeaned ages
    ("xreduce", self.x_reduced), # HIGH — PCA-reduced features
    ...
]
```

`save_text()` writes the raw training data, normalised features, and PCA-reduced features to disk in human-readable format. This function should be disabled or restricted in a TRE deployment — it creates files that could be inadvertently or deliberately transmitted through the egress gateway.

**Required control:** Remove or gate `save_text()` in TRE deployments. If needed for debugging, the output files must be classified as restricted data and subject to the same egress controls as raw data.

---

## Logging Risks

```python
# deltab/deltab.py line 75
LOG.info(f" - Number of subjects: {self.num_subjects_train}, number of features: {self.num_features_train}")

# deltab/main.py lines 46-48
LOG.debug(f" - Removing subject {idx} because NaN found in age or features")
LOG.info(f" - Removed {num_subjects-len(ages_out)} subjects with NaN")
```

Log output can leak:

- Exact subject count N (sensitive in small TREs — see [Thresholds](thresholds.md))
- The number of subjects removed due to NaN (could indicate specific data quality issues)
- At DEBUG level, the index of each removed subject

**Required control:** In TRE deployments, set logging to WARNING or above. Log files must be treated as restricted outputs and not egressed without review. Debug-level logging must be disabled by default.

---

## NaN Imputation Risk

```python
# deltab/main.py lines 51-59
def _nan_median_impute(features):
    median = np.nanmedian(features, axis=1)
    ...
```

Median imputation computes per-feature medians from the local cohort. These medians are not stored as attributes and are not transmitted directly. However, if a central server requests imputation statistics (e.g. to perform imputation centrally), these are population-level summaries that require SDC review.

---

## Recommended Egress Controls Summary

| Output type | Egress decision | Control required |
|-------------|----------------|------------------|
| `save()` pickle file | **BLOCK** | Never egress — contains raw subject data |
| `save_text()` output folder | **BLOCK** | Never egress — contains raw features and ages |
| Model weights only (b1, b2, PCA) | **REVIEW** | SDC review; check N ≥ threshold; check no inversion possible |
| Per-subject delta vector | **REVIEW** | SDC review; suppression/aggregation required |
| Aggregated delta statistics (mean, SD) | **CONDITIONAL** | N ≥ minimum cell size; no small subgroups |
| Log files | **REVIEW** | Check for subject counts and indices before egress |
| N (number of subjects) | **CONDITIONAL** | Only if N ≥ minimum disclosure threshold |

---

## Code Modifications Required for TRE Deployment

The following changes are necessary before deltab can be deployed in a DPUK or similar TRE context. None of these are currently implemented.

1. **Implement `save_weights_only()`** — serialise only aggregate model parameters, explicitly excluding `xtrain`, `ytrain`, `x_norm`, `x_reduced`, `y_demean`, `ysq*`, `y2*`.

2. **Disable or restrict `save_text()`** — gate behind a TRE deployment flag, or remove entirely from the TRE-targeted build.

3. **Scrub subject-level data post-training** — after `train()` completes, optionally zero or delete `xtrain`, `ytrain`, `x_norm`, `x_reduced` from the object to prevent accidental serialisation.

4. **Enforce minimum N check** — raise an exception or warning if `num_subjects_train` falls below the minimum disclosure threshold (suggested: N < 10 for initial output; N < 20 for subgroup analysis).

5. **Restrict logging** — default log level to WARNING in TRE mode; ensure log files are flagged as restricted outputs.

6. **Add SACRO-compatible output metadata** — tag all egress-eligible outputs with provenance metadata (node ID, N, model version, date) to support central audit trail.

---

## References

- DPUK Data Access and Governance Framework. Dementias Platform UK. [https://www.dementiasplatform.uk](https://www.dementiasplatform.uk)
- TREvolution SACRO Project. Statistical Analysis for Controlled Release of Outputs. [https://github.com/AI-SDC/SACRO](https://github.com/AI-SDC/SACRO)
