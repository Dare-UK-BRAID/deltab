# Command Line Interface

The `deltab` command provides a full brain age estimation workflow from the terminal. It can train a model from scratch, load a previously saved model, apply any of the five prediction models, and write outputs to text files.

## Basic usage

```bash
deltab [OPTIONS]
```

Running `deltab` with no arguments prints the help summary.

---

## Complete argument reference

### Input / model loading

| Argument | Type | Description |
|----------|------|-------------|
| `--train-ages FILE` | path | Delimited text file containing one true age per subject (1D). Space, comma, or tab delimited. Header rows are handled automatically. |
| `--train-features FILE` | path | Delimited text file containing the feature matrix (subjects × features, 2D). One row per subject, one column per feature. |
| `--load FILE` | path | Skip training and load a previously saved model from a pickle file. Cannot be combined with `--train-ages` / `--train-features`. |

### Feature pre-processing

| Argument | Default | Description |
|----------|---------|-------------|
| `--feature-nans {median,remove}` | `median` | Strategy for subjects who have NaN in one or more features. `median` replaces each NaN with the median value of that feature across all subjects. `remove` drops those subjects entirely from training. |
| `--feature-proportion FLOAT` | — | Proportion (0–1) of PCA components to retain. E.g. `0.1` retains the top 10% of components. Mutually exclusive with `--feature-num`, `--feature-var`, and `--kaiser-guttmann`. |
| `--feature-num INT` | — | Exact number of PCA components to retain. |
| `--feature-var FLOAT` | — | Retain components that collectively explain at least this proportion of total variance (0–1). |
| `--kaiser-guttmann` | off | Select PCA components using the Kaiser-Guttmann criterion (retain components whose eigenvalue exceeds the mean eigenvalue). |

!!! note "No PCA reduction"
    If none of the `--feature-*` options are specified, all features are used without PCA reduction. This is only appropriate when the number of subjects substantially exceeds the number of features (N >> F).

### Prediction

| Argument | Default | Description |
|----------|---------|-------------|
| `--predict {age,delta}` | `delta` | Output mode. `age` outputs the predicted brain age (years). `delta` outputs the brain age delta (predicted age minus true age, in years). |
| `--predict-model MODEL` | `unbiased_quadratic` | Which model to use. One of `simple`, `unbiased`, `unbiased_quadratic`, `alternate`, `alternate_quadratic`. See [Prediction Models](models.md). |
| `--predict-output FILE` | `deltab.txt` | Path to write the prediction results to. One value per subject, one per line. |
| `--predict-ages FILE` | — | True ages for prediction subjects. Only needed when predicting on a different set of subjects from those used in training. Must be provided together with `--predict-features`. |
| `--predict-features FILE` | — | Feature matrix for prediction subjects. Only needed when predicting on different subjects. |
| `--true-ages-output FILE` | — | Write the true ages of subjects actually included in the prediction to this file. Useful when `--feature-nans remove` may have dropped some subjects, to keep ages aligned with predicted deltas. |

### Model saving

| Argument | Description |
|----------|-------------|
| `--save FILE` | Save the trained model to a pickle file for later reuse with `--load`. |
| `--save-text DIR` | Save the trained model data (feature matrix, PCA components, age vectors) as plain text files in a directory. Useful for inspection or loading into other analysis tools. Note: there is currently no `--load` facility for text-format models. |

### General

| Argument | Description |
|----------|-------------|
| `--overwrite` | Overwrite the output file if it already exists. Without this flag, deltab will error rather than silently overwrite results. |
| `--debug` | Enable verbose debug logging to stderr. |
| `--version` / `-v` | Print the version string and exit. |

---

## Examples

### Train and predict on the same cohort

```bash
deltab \
  --train-ages ages.txt \
  --train-features idps.txt \
  --feature-proportion 0.1 \
  --predict-model unbiased_quadratic \
  --predict delta \
  --predict-output brain_age_delta.txt
```

This is the most common usage: train on a cohort, apply the unbiased quadratic model, and write the brain age delta for each subject.

### Train and save the model, predict on a new cohort

```bash
# Step 1: train on the normative cohort and save the model
deltab \
  --train-ages normative_ages.txt \
  --train-features normative_idps.txt \
  --feature-proportion 0.15 \
  --save trained_model.pkl \
  --predict age \
  --predict-output /dev/null

# Step 2: apply the saved model to a new clinical cohort
deltab \
  --load trained_model.pkl \
  --predict-ages clinical_ages.txt \
  --predict-features clinical_idps.txt \
  --predict-model unbiased_quadratic \
  --predict delta \
  --predict-output clinical_delta.txt
```

### Handle subjects with missing features

```bash
# Remove subjects with any NaN and record which subjects were kept
deltab \
  --train-ages ages.txt \
  --train-features idps.txt \
  --feature-nans remove \
  --true-ages-output ages_included.txt \
  --predict delta \
  --predict-output delta.txt
```

`ages_included.txt` will contain only the ages of subjects that were not removed, keeping the age vector aligned with the delta output.

### Use Kaiser-Guttmann component selection

```bash
deltab \
  --train-ages ages.txt \
  --train-features idps.txt \
  --kaiser-guttmann \
  --predict delta \
  --predict-output delta_kg.txt
```

---

## Input file format

Both `--train-ages` / `--predict-ages` and `--train-features` / `--predict-features` accept text files that are:

- Space, comma, or tab delimited
- Optionally quoted (single or double)
- Optionally with a single header row (auto-detected)

The age file should contain a single column of numeric values. The feature file should be a 2D matrix with subjects as rows and features as columns. Column headers, if present, are ignored.

---

## Output file format

Predicted ages or deltas are written as plain text, one value per line, with no header. The number of lines equals the number of subjects in the prediction set. Values are written using NumPy's `savetxt` with default formatting (up to 18 significant figures).
