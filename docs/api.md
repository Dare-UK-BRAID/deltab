# Python API

The `deltab` package exposes two public objects: the `BrainDelta` class and the `Model` constants.

```python
from deltab import BrainDelta, Model
```

---

## `Model` constants

`Model` defines the five prediction models as integer class attributes.

| Constant | Value | CLI equivalent |
|----------|-------|---------------|
| `Model.SIMPLE` | 1 | `simple` |
| `Model.UNBIASED` | 2 | `unbiased` |
| `Model.UNBIASED_QUADRATIC` | 3 | `unbiased_quadratic` |
| `Model.ALTERNATE` | 4 | `alternate` |
| `Model.ALTERNATE_QUADRATIC` | 5 | `alternate_quadratic` |

Convert a string to a constant with `Model.fromstr()`:

```python
model = Model.fromstr("unbiased_quadratic")  # returns Model.UNBIASED_QUADRATIC
```

---

## `BrainDelta` class

`BrainDelta` is the main class. A new instance is untrained. Call `train()` before `predict()`.

```python
b = BrainDelta()
```

---

### `train()`

Train the brain age model on a set of subjects with known ages and features.

```python
b.train(
    ages,
    features,
    ev_proportion=None,
    ev_num=None,
    ev_kg=False,
    ev_var=None,
    conf=None
)
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ages` | array-like, shape (N,) | True chronological ages for N training subjects. Must be 1D. |
| `features` | array-like, shape (N, F) | Feature matrix. N subjects, F features. Each row is one subject. |
| `ev_proportion` | float or None | Proportion (0–1) of PCA components to retain. Mutually exclusive with `ev_num`, `ev_kg`, `ev_var`. |
| `ev_num` | int or None | Exact number of PCA components to retain. |
| `ev_kg` | bool | If `True`, use Kaiser-Guttmann criterion to select the number of components. Default `False`. |
| `ev_var` | float or None | Retain components that explain at least this proportion of total variance. |
| `conf` | array-like, shape (N, C) or None | Optional confound matrix. C confounding variables (e.g. scanner site, sex as dummy-coded columns) whose effect is regressed out of the feature matrix before model fitting. |

**Notes**

- At most one of `ev_proportion`, `ev_num`, `ev_kg`, `ev_var` may be specified. Passing more than one raises `ValueError`.
- If none are specified, all features are used without PCA reduction.
- The training means, standard deviations, and PCA components are stored internally and reused automatically during `predict()`.

**Example**

```python
import numpy as np
from deltab import BrainDelta

ages = np.loadtxt("ages.txt")           # shape (500,)
features = np.loadtxt("idps.txt")       # shape (500, 1500)

b = BrainDelta()
b.train(ages, features, ev_proportion=0.1)
```

---

### `predict()`

Predict brain age or brain age delta for a set of subjects.

```python
result = b.predict(
    age,
    features,
    model=Model.UNBIASED_QUADRATIC,
    return_delta=False,
    conf=None
)
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `age` | array-like, shape (N,) | True chronological ages for N prediction subjects. |
| `features` | array-like, shape (N, F) | Feature matrix. F must match the number of features used in training. |
| `model` | `Model` constant | Which model to use. Default `Model.UNBIASED_QUADRATIC`. |
| `return_delta` | bool | If `True`, return brain age delta (predicted age − true age). If `False` (default), return predicted brain age in years. |
| `conf` | array-like or None | Confound matrix for prediction subjects, matching the confounds used in training. Required if `conf` was passed to `train()`. |

**Returns**

Array of shape (N,) containing predicted brain ages or brain age deltas.

**Raises**

- `RuntimeError` if the model has not been trained.
- `ValueError` if the number of features does not match training, or if `conf` is inconsistently provided.

**Example**

```python
# Predict delta on the same subjects used for training
delta = b.predict(ages, features, model=Model.UNBIASED_QUADRATIC, return_delta=True)

# Predict brain age (not delta) on new subjects
new_ages = np.loadtxt("new_ages.txt")
new_features = np.loadtxt("new_idps.txt")
brain_age = b.predict(new_ages, new_features, return_delta=False)
```

---

### `save()`

Serialise the trained model to a file using Python's `pickle` module.

```python
b.save(fname)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `fname` | str or path | File path to write to. Overwrites without warning. |

The saved file contains the entire `BrainDelta` object, including training data, PCA components, and all fitted coefficients.

!!! warning "Pickle security"
    Never load a pickle file from an untrusted source. The `pickle` format can execute arbitrary code on load.

---

### `load()`

Load a previously saved model.

```python
b.load(fname)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `fname` | str or path | Path to a pickle file saved by `save()`. |

After loading, the object is ready for `predict()` without retraining.

```python
b = BrainDelta()
b.load("trained_model.pkl")
delta = b.predict(ages, features, return_delta=True)
```

---

### `save_text()`

Save the trained model data to a directory as plain text files. Useful for inspection, debugging, or loading into MATLAB or R.

```python
b.save_text(dpath)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `dpath` | str or path | Directory to create and write files into. Created if it does not exist. |

**Output files**

| File | Content | Shape |
|------|---------|-------|
| `ytrain` | True ages used in training | (N,) |
| `xtrain` | Raw feature matrix used in training | (N, F) |
| `ynorm` | Demeaned age vector (Ỹ) | (N,) |
| `xnorm` | Normalised feature matrix (X̃) | (N, F) |
| `xreduce` | PCA-reduced feature matrix (X_r) | (N, J) |
| `ysq` | Squared demeaned age | (N,) |
| `ysqnorm` | Demeaned squared age | (N,) |
| `ysqorth` | Orthogonalised squared age (Ỹ²ₒ) | (N,) |
| `pca_components` | PCA component matrix | (J, F) — only if PCA was used |
| `pca_explained_variance` | Explained variance per component | (J,) — only if PCA was used |

!!! note "No load facility for text format"
    There is currently no corresponding `load_text()` method. Use `save()` / `load()` (pickle) for model persistence in workflows.

---

## Full workflow example

```python
import numpy as np
from deltab import BrainDelta, Model

# --- Training ---
ages_train    = np.loadtxt("normative_ages.txt")     # (N_train,)
features_train = np.loadtxt("normative_idps.txt")    # (N_train, F)

b = BrainDelta()
b.train(ages_train, features_train, ev_proportion=0.15)
b.save("normative_model.pkl")

# --- Prediction on new cohort ---
b2 = BrainDelta()
b2.load("normative_model.pkl")

ages_test    = np.loadtxt("clinical_ages.txt")       # (N_test,)
features_test = np.loadtxt("clinical_idps.txt")      # (N_test, F)

delta = b2.predict(ages_test, features_test,
                   model=Model.UNBIASED_QUADRATIC,
                   return_delta=True)

np.savetxt("clinical_brain_age_delta.txt", delta)
print(f"Mean delta: {np.mean(delta):.2f} years")
print(f"Std delta:  {np.std(delta):.2f} years")
```

---

## Handling NaN values

The CLI handles NaN values automatically via `--feature-nans`. When using the Python API directly, apply the same logic manually before calling `train()`:

```python
# Option 1: Remove subjects with any NaN
valid = ~np.any(np.isnan(features), axis=1)
ages_clean    = ages[valid]
features_clean = features[valid]

# Option 2: Impute median per feature
median = np.nanmedian(features, axis=0)
inds = np.where(np.isnan(features))
features[inds] = np.take(median, inds[1])
```
