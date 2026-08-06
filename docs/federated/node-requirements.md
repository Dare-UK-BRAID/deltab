# Node Data & Software Requirements

This page specifies what must be present at each federated TRE node to run deltab for brain age estimation, covering input data formats, temporary data created during processing, software dependencies, and data lifecycle controls.

---

## Input Data Requirements

### 1. Subject age file

| Property | Specification |
|----------|--------------|
| Format | Plain text, one value per line (`.txt`) or single-column CSV |
| Content | True chronological age in years (decimal, e.g. `67.4`) |
| Shape | N × 1 where N = number of subjects |
| Acceptable range | Typically 40–90 years for dementia cohorts; no hard limit in code |
| Missing values | NaN accepted; subjects with NaN age are removed by `_remove_nan_subjects()` in `main.py` |
| Classification | **RESTRICTED** — chronological ages are quasi-identifying data; subject to full TRE data access controls |
| Storage location | Must remain within the TRE secure enclave at all times |

### 2. Feature matrix (IDPs)

| Property | Specification |
|----------|--------------|
| Format | Plain text, space/tab/comma delimited (`.txt` or `.csv`) |
| Content | One row per subject, one column per imaging-derived phenotype (IDP) |
| Shape | N × F where F = number of features (typically 50–2000 IDPs from BRC Pipeline) |
| Acceptable values | Real-valued; NaN permitted (imputed or subject removed depending on `--feature-nans` flag) |
| Typical IDP sources | BRC structural pipeline (cortical thickness, subcortical volumes, SIENAX), diffusion pipeline (FA, MD, TBSS), FLAIR WMH load |
| Missing values | Default: median imputation per feature via `_nan_median_impute()`. Alternative: subject removal |
| Classification | **RESTRICTED** — IDP vectors are potentially re-identifying; must remain within TRE |
| Storage location | Must remain within the TRE secure enclave |

### 3. Optional: confound matrix

| Property | Specification |
|----------|--------------|
| Format | Same as feature matrix (plain text, N × C) |
| Content | Confounding variables to be regressed out of features before model fitting (e.g. scanner site, sex as binary, head motion) |
| Shape | N × C where C = number of confounds |
| Classification | **RESTRICTED** — confounds may include demographic data |
| When required | Optional; only needed if scanner site or sex confounds are present and should be removed |

### 4. Pre-trained model file (inference mode only)

| Property | Specification |
|----------|--------------|
| Format | Python pickle file (`.pkl`) produced by `BrainDelta.save()` |
| Content | Serialised `BrainDelta` object — **see Egress Risk Analysis** |
| Classification | **RESTRICTED if trained on local data** — contains raw training arrays (see [Egress page](egress.md)). Weights-only exports (once implemented) would be LOW risk |
| Source | Either trained locally or received from central aggregator (weights only) |

---

## Temporary Data Created During Processing

During a deltab training and inference run, the following intermediate data are created in memory and optionally written to disk:

### In-memory (never written to disk unless explicitly requested)

| Variable | Content | Risk | Lifecycle |
|----------|---------|------|-----------|
| `self.x_norm` | Normalised IDP matrix (N × F) | HIGH | Held in RAM during training; persists in object until garbage collected or `save()` called |
| `self.x_reduced` | PCA-reduced features (N × J) | HIGH | As above |
| `self.y_demean` | Demeaned ages (N,) | HIGH | As above |
| `self.ysq`, `self.ysq_orth` | Quadratic age terms (N,) | MEDIUM | As above |
| `features_norm` (predict) | Normalised test features | HIGH | Local variable in `predict()`; freed after return |

### Written to disk (only if explicitly requested)

| File / folder | Created by | Content | Risk |
|---------------|-----------|---------|------|
| `<save_path>.pkl` | `BrainDelta.save()` | Full object pickle — includes raw training data | **HIGH** — must be classified as RESTRICTED and not egressed |
| `<save_text_path>/ytrain` | `BrainDelta.save_text()` | Raw training ages | **HIGH** |
| `<save_text_path>/xtrain` | `BrainDelta.save_text()` | Raw training IDP matrix | **HIGH** |
| `<save_text_path>/xnorm` | `BrainDelta.save_text()` | Normalised IDPs | **HIGH** |
| `<save_text_path>/ynorm` | `BrainDelta.save_text()` | Demeaned ages | **HIGH** |
| `<save_text_path>/xreduce` | `BrainDelta.save_text()` | PCA-reduced IDPs | **HIGH** |
| `<save_text_path>/pca_components` | `BrainDelta.save_text()` | PCA eigenvectors | **MEDIUM** |
| `<predict_output>.txt` | `main.py` | Per-subject predicted deltas or ages | **HIGH** — must not be egressed without SDC review and DAC approval |
| Log output (stderr) | Logging module | Subject counts, removed subjects (at DEBUG level) | **MEDIUM** |

!!! warning "Temporary file controls"
    All files written to disk during a deltab run must be stored within the TRE secure enclave. Automatic cleanup of temporary files after the run is strongly recommended. The `save()` pickle and `save_text()` outputs should be classified as RESTRICTED data and subject to the same controls as raw subject data.

---

## Software Requirements

### Python environment

| Package | Minimum version | Purpose |
|---------|----------------|---------|
| Python | 3.8 | Runtime |
| numpy | 1.20 | Array operations, pseudoinverse, PCA |
| scikit-learn | 0.24 | `sklearn.decomposition.PCA` |
| pandas | 1.2 | Used in example scripts (table generation) |
| diffprivlib | 0.6 | **Recommended addition** for DP-PCA and DP linear regression |

### Installation in TRE environments

TRE nodes typically have no outbound internet access. All packages must be pre-installed or supplied as a local mirror:

```bash
# Option 1: Pre-install in a virtual environment before network isolation
python -m venv deltab_env
source deltab_env/bin/activate
pip install deltab numpy scikit-learn pandas

# Option 2: Use a pre-built wheel bundle
pip install --no-index --find-links=/path/to/local/wheels deltab
```

For HPC clusters (e.g. DPUK SeRP compute nodes), a Singularity/Apptainer container image is recommended to ensure reproducibility and avoid dependency conflicts with the node's system Python.

### No network access required

deltab has no runtime network dependencies. All computation is local. This is consistent with TRE offline compute environments.

---

## Data Lifecycle Controls

| Stage | Control |
|-------|---------|
| Data ingestion | Input files copied to TRE secure enclave; original source data not duplicated outside approved storage |
| Processing | All intermediate arrays in memory; no swap-to-disk in TRE environments (disable paging or use encrypted swap) |
| Output staging | All output files written to a designated restricted output area, not accessible to the standard file egress gateway |
| Egress review | SACRO or equivalent SDC tool reviews all staged outputs before release |
| Cleanup | After egress decision, all temporary and output files not approved for egress are securely deleted from the staging area |
| Audit trail | Logging of run parameters (N, J, model type, date) maintained in the TRE audit log — not transmitted externally |
