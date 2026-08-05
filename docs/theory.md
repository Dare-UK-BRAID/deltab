# Theory — Brain Age Estimation

This page describes the mathematical foundations of the brain age estimation framework implemented in deltab, based on Smith et al. (2019).

---

## Notation

| Symbol | Meaning |
|--------|---------|
| **Y** | Vector of true chronological ages, shape (N,) |
| **X** | Matrix of brain imaging features, shape (N, F) |
| **Ỹ** | Demeaned age vector: Ỹ = Y − mean(Y) |
| **X̃** | Normalised feature matrix: zero mean, unit std per column |
| **X_r** | PCA-reduced feature matrix, shape (N, J) |
| **δ₁** | Initial (biased) brain age delta |
| **δ₂** | Corrected brain age delta (unbiased model) |
| **δ₂q** | Corrected brain age delta with quadratic correction |
| N | Number of subjects |
| F | Number of features |
| J | Number of PCA components retained |
| **β₁** | Regression coefficients from features to age |
| **β₂** | Correction coefficients |

---

## The Algorithm

The deltab implementation follows the algorithm described in Smith et al. (2019). The steps below correspond directly to the numbered steps in the module docstring in `deltab/deltab.py`.

### Step 1 — Assemble inputs

Collect the age vector **Y** (shape N×1) and the feature matrix **X** (shape N×F). Features are typically IDPs from a neuroimaging pipeline, but can be any set of brain-derived measures — voxel intensities, connectivity matrices, or aggregated scalars.

### Step 2 — Normalise

Subtract the mean from **Y** to obtain demeaned age **Ỹ**. Standardise each column of **X** to zero mean and unit standard deviation to produce **X̃**. Normalisation ensures that no single feature dominates the regression due to its scale rather than its predictive power.

```python
# From deltab.py — BrainDelta.train()
self.y_mean = np.mean(self.ytrain)
self.y_demean = self._standardize_y(self.ytrain)          # Ỹ = Y - mean(Y)
self.x_norm, self.x_mean, self.x_std = self._normalize(self.xtrain)  # X̃
```

### Step 3 — Optional deconfounding

If confounding variables (e.g. scanner site, sex) are passed via the `conf` argument, their effect is regressed out of **X̃** before PCA and model fitting:

$$\tilde{X} \leftarrow \tilde{X} - C \cdot (C^+ \tilde{X})$$

where **C** is the confound matrix and **C⁺** is its pseudoinverse. The same confound removal is applied at prediction time.

### Step 4 — PCA dimensionality reduction

Brain imaging datasets typically have far more features than subjects (F >> N). Fitting a regression directly on **X̃** is ill-conditioned. PCA reduces the feature space to the top J components, retaining the dominant axes of variation while discarding noise.

deltab wraps scikit-learn's `PCA` class and supports four ways to select J:

| Option | Parameter | Description |
|--------|-----------|-------------|
| Fixed proportion | `ev_proportion` | Retain a fraction (0–1) of columns, e.g. 0.1 = top 10% |
| Fixed count | `ev_num` | Retain exactly J components |
| Variance explained | `ev_var` | Retain components until this fraction of total variance is explained |
| Kaiser-Guttmann | `ev_kg` | Retain components with eigenvalue > mean eigenvalue |
| No reduction | *(none)* | Use all features (only suitable when N >> F) |

Smith et al. recommend retaining 10–25% of components. The Kaiser-Guttmann criterion is a data-driven alternative that retains components whose variance exceeds the average, analogous to retaining eigenvalues > 1 in factor analysis.

```python
# From deltab.py
elif ev_kg:
    pca = PCA()
    pca.fit(self.x_norm)
    mean = np.mean(pca.explained_variance_)
    ev_num_kg = len([v for v in pca.explained_variance_ if v > 1])
    self.pca = PCA(n_components=ev_num_kg)
```

The PCA fit is performed on the **training data only**. When predicting on new subjects, the saved PCA transform is applied to their features, ensuring no information from test subjects leaks into the model.

### Step 5 — Quadratic age term

To model non-linear (quadratic) dependence of brain features on age, a squared age term is constructed. Simply squaring **Ỹ** would be collinear with **Ỹ** for symmetric age distributions, so the squared term is first demeaned and then **orthogonalised** with respect to **Ỹ**:

$$\tilde{Y}^2_o = \tilde{Y}^2 - \text{mean}(\tilde{Y}^2) - \left(\frac{\tilde{Y}}{||\tilde{Y}||} \cdot \tilde{Y}^2_\text{demean}\right) \cdot \tilde{Y}$$

This ensures the quadratic component captures only the truly quadratic variation, not the linear component already captured by **Ỹ**.

```python
# From deltab.py
self.ysq_orth = self._orthogonalize(self.ysq_demean, self.y_demean)
```

The combined age matrix for quadratic models is then **Y₂ = [Ỹ, Ỹ²ₒ]**, shape N×2.

### Step 6 — Initial age prediction (β₁)

The reduced feature matrix **X_r** is used to predict demeaned age **Ỹ** via ordinary least squares using the Moore-Penrose pseudoinverse:

$$\hat{Y}_{B1} = X_r \cdot \beta_1, \quad \text{where } \beta_1 = X_r^+ \cdot \tilde{Y}$$

The initial brain age delta is:

$$\delta_1 = \hat{Y}_{B1} - \tilde{Y}$$

This is what the `simple` model returns (after adding back the age mean). It is systematically biased: subjects at the extremes of the age distribution have artificially large or small deltas because regression pulls all predictions towards the mean.

```python
# From deltab.py
self.b1 = np.dot(np.linalg.pinv(self.x_reduced), self.y_demean)  # β₁
y_b1 = np.dot(self.x_reduced, self.b1)                            # Ŷ_B1
d1 = y_b1 - self.y_demean                                         # δ₁
```

### Step 7 — Bias correction (β₂)

The bias in δ₁ correlates with true age **Ỹ**. To remove it, δ₁ is regressed onto the age matrix **Y₂** and the fitted component is subtracted:

$$\beta_2 = Y_2^+ \cdot \delta_1$$
$$\delta_2 = \delta_1 - Y_2 \cdot \beta_2$$

For the linear-only case (`unbiased`), **Y₂ = Ỹ** (a column vector). For the quadratic case (`unbiased_quadratic`), **Y₂ = [Ỹ, Ỹ²ₒ]**, which additionally removes any systematic quadratic relationship between delta and true age.

```python
# From deltab.py
self.b2   = np.dot(np.linalg.pinv(self.y2),   d1)   # linear correction
self.b2sq = np.dot(np.linalg.pinv(self.y2sq), d1)   # quadratic correction
```

---

## The Bias Problem in Detail

The Smith et al. paper demonstrates that the simple regression model produces biased deltas under several common real-world conditions:

- **Age range truncation** — if the training cohort spans only part of the natural age range (common in TRE datasets which may have age eligibility criteria), regression to the mean is exacerbated.
- **Regression dilution** — measurement noise on the features attenuates the slope of the fitted model, increasing bias.
- **Noise on the predicted variable** — additional variance in the target (age) beyond what is captured by the model inflates deltas.
- **Retained age dependence in features** — if features are not fully age-normalised before prediction (the common case), the simple model retains a correlation between predicted age and true age that manifests as biased delta.

The corrected models (unbiased, unbiased_quadratic, alternate, alternate_quadratic) all address the bias at different levels of complexity. See [Prediction Models](models.md) for a comparison.

---

## Prediction on New Subjects

When predicting brain age for subjects not in the training set, the same normalisation (using training means and standard deviations) and the same PCA transform are applied to the new subjects' features. This is critical: using the test subjects' own statistics for normalisation would constitute data leakage.

```python
# From deltab.py — BrainDelta._standardize_x()
def _standardize_x(self, x, conf=None):
    x_norm = self._normalize(x, mean=self.x_mean, std=self.x_std)[0]  # uses training mean/std
    if self.pca is not None:
        features_norm = self.pca.transform(features_norm)              # uses training PCA
```

The age standardisation at prediction time uses the **training mean** (i.e. subjects' ages are demeaned relative to the training cohort mean, not their own cohort mean):

```python
def _standardize_y(self, y):
    return y - self.y_mean  # self.y_mean is fixed from training
```

This means that if you train on a cohort with mean age 60 and predict on a cohort with mean age 70, the age standardisation will still subtract 60 from each prediction subject's age. This is intentional: the model's internal coordinate system is fixed to the training data, and deviations from that reference are what define the brain age delta.

---

## References

- Smith, S.M. et al. (2019). Estimation of brain age delta from brain imaging. *NeuroImage*, 200, 528–539. [https://doi.org/10.1016/j.neuroimage.2019.06.017](https://doi.org/10.1016/j.neuroimage.2019.06.017)
- Franke, K. & Gaser, C. (2019). Ten years of BrainAGE as a neuroimaging biomarker of brain aging. *Frontiers in Neurology*, 10, 789. [https://doi.org/10.3389/fneur.2019.00789](https://doi.org/10.3389/fneur.2019.00789)
- Le, T.T. et al. (2018). A nonlinear simulation framework supports adjusting for age when removing brain age effects. *Frontiers in Aging Neuroscience*, 10, 317. [https://doi.org/10.3389/fnagi.2018.00317](https://doi.org/10.3389/fnagi.2018.00317)
