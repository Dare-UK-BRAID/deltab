# Membership Inference Attacks

Membership inference attacks (MIAs) attempt to determine whether a specific individual's data was used to train a model, using only the model's outputs or parameters. For brain age models trained on TRE cohorts, a successful MIA could confirm that a named individual was part of a study — a disclosure in itself, regardless of whether any clinical information is revealed.

---

## Why Brain Age Models Are Vulnerable

Brain age models trained with deltab expose several properties that facilitate membership inference:

**1. Regression residuals as membership signals**

The `simple` model produces predictions that are systematically biased towards the training mean. Subjects whose data was in the training set tend to have smaller residuals (the model has partially "memorised" them) than held-out subjects. An attacker with access to the model weights and a subject's IDP vector can compute the residual and use its magnitude as a membership signal.

This is particularly acute when:

- The training cohort is small (N < 500 at a TRE node)
- PCA retains many components (high J), increasing the model's capacity to overfit
- The cohort has a distinctive age distribution (e.g. a disease-enriched registry)

**2. PCA component leakage**

The exported PCA components (`pca.components_`) encode the covariance structure of the training feature matrix. An attacker who knows a subject's IDP vector can project it onto the PCA components and measure how well it aligns with the training data's principal axes. Strong alignment is a positive membership signal.

**3. Normalisation statistics**

The exported `x_mean` and `x_std` vectors reflect the training cohort's feature distribution. An adversary can compare a candidate subject's features against these statistics to infer whether the subject's data would have been within the typical range seen during training.

**4. Age mean leakage**

`y_mean` (the scalar training age mean) combined with `b2` (the bias correction slope) can be used to infer the approximate age range of the training cohort — information that narrows which subjects could have been in the training set.

---

## Attack Scenarios

### Scenario 1 — Black-box inference via prediction API

An attacker submits a target subject's IDP vector to a federated deltab inference endpoint and observes the returned brain age delta. The magnitude and sign of the delta, compared against population norms, is used as a membership signal.

**Risk level:** Moderate. The signal is noisy because brain age delta varies substantially between individuals, but repeated queries can narrow uncertainty.

**Control:** Rate-limit or audit inference API calls. Do not return per-subject deltas without SDC review. Return only aggregated group statistics.

### Scenario 2 — White-box inference from exported model weights

An attacker obtains the exported model weights (β₁, PCA components, normalisation statistics) — either legitimately as a collaborator or through a data breach — and uses them to run a shadow model attack. They train shadow models on synthetic data, learn a meta-classifier, and apply it to determine membership for target subjects.

**Risk level:** High when N is small and J is large. Shadow model attacks have been shown to achieve >80% accuracy against regression models trained on fewer than 200 subjects. [Citation Needed]

**Control:** Apply differential privacy noise to exported weights (see [Differential Privacy](differential-privacy.md)). Limit J (number of PCA components) to reduce model capacity. Enforce N ≥ 100 minimum per node before training is permitted.

### Scenario 3 — Reconstruction via PCA inversion

Given the PCA components and per-subject delta, an attacker attempts to reconstruct the approximate IDP feature vector for a target subject by inverting the PCA projection. Reconstructed features can be compared against known databases (e.g. population normative IDP distributions) to infer identity.

**Risk level:** Moderate to High depending on J and the uniqueness of the target's IDP profile.

**Control:** Do not export `pca.components_` without review. Use aggregate PCA loadings across nodes rather than node-specific components. Apply output perturbation before egress.

### Scenario 4 — Differential membership via cohort statistics

An attacker knows the mean and SD of brain age delta for a TRE cohort (these are low-risk summary statistics that would commonly be approved for egress). They test whether a specific individual's predicted delta is consistent with membership by comparing it against the published distribution.

**Risk level:** Low for healthy population cohorts; moderate for disease-enriched cohorts where delta distributions are distinctive.

**Control:** Publish only coarse summary statistics. Avoid publishing full delta distributions for small subgroups (see [Disclosure Thresholds](thresholds.md)).

---

## deltab-Specific Vulnerabilities

Reviewing the source code directly:

```python
# deltab/deltab.py line 135
self.b1 = np.dot(np.linalg.pinv(self.x_reduced), self.y_demean)
```

**β₁** is fitted to minimise the sum of squared residuals on the training data. A subject whose IDP vector was in `x_reduced` during training will produce a smaller prediction residual than a non-member subject with a similar IDP profile. This residual gap is the core membership signal.

```python
# deltab/deltab.py lines 143-144
self.b2 = np.dot(np.linalg.pinv(self.y2), d1)
self.b2sq = np.dot(np.linalg.pinv(self.y2sq), d1)
```

**β₂** encodes the age-delta correlation in the training data. If a target subject's age-delta pair aligns closely with the β₂ correction slope, this is evidence of membership.

```python
# deltab/deltab.py line 113
self.x_reduced = self.pca.fit_transform(self.x_norm)
```

The PCA is fitted on training data. The projection of a training subject's features will have lower reconstruction error (‖x − PCA⁻¹(PCA(x))‖) than a non-training subject, providing another membership signal.

---

## Mitigations

| Attack vector | Mitigation | Implementation |
|--------------|-----------|----------------|
| Residual-based inference | Limit prediction API access; return group deltas only | Policy and API design |
| PCA component leakage | Add Gaussian noise to exported components (DP); aggregate across nodes | `pca.components_ += np.random.normal(0, σ, ...)` |
| Shadow model attack | Enforce minimum N; apply DP to weights | N ≥ 100 per node; DP noise on β₁ |
| Reconstruction attack | Do not export node-specific PCA components | Policy; export only cross-node aggregate PCA |
| Cohort distribution inference | Suppress small-subgroup statistics | N ≥ minimum cell size (see Thresholds) |
| Log-based inference | Restrict and audit log output | WARNING-level logging only |

---

## References

- Shokri, R. et al. (2017). Membership Inference Attacks Against Machine Learning Models. *IEEE S&P*. [https://doi.org/10.1109/SP.2017.41](https://doi.org/10.1109/SP.2017.41)
- Yeom, S. et al. (2018). Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting. *IEEE CSF*. [https://doi.org/10.1109/CSF.2018.00027](https://doi.org/10.1109/CSF.2018.00027)
- Kaissis, G. et al. (2021). End-to-end privacy preserving deep learning on multi-institutional medical imaging. *Nature Machine Intelligence*, 3, 473–484. [https://doi.org/10.1038/s42256-021-00337-8](https://doi.org/10.1038/s42256-021-00337-8)
- Fredrikson, M. et al. (2015). Model Inversion Attacks that Exploit Confidence Information and Basic Countermeasures. *ACM CCS*. [https://doi.org/10.1145/2810103.2813677](https://doi.org/10.1145/2810103.2813677)
