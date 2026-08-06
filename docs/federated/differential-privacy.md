# Differential Privacy

Differential Privacy (DP) provides a mathematical guarantee that the output of a computation does not change meaningfully whether or not any single individual's data is included in the input. For federated brain age estimation, DP offers a principled mechanism to protect training subjects from membership inference and model inversion attacks while still allowing aggregate model parameters to be shared with a central aggregator.

---

## Formal Definition

A randomised mechanism **M** satisfies **(ε, δ)-differential privacy** if for any two datasets **D** and **D'** differing in one subject's record, and for any set of outputs S:

> Pr[**M**(D) ∈ S] ≤ e^ε · Pr[**M**(D') ∈ S] + δ

where:

- **ε** (epsilon) is the privacy budget — smaller values give stronger privacy guarantees.
- **δ** is the probability of a catastrophic privacy failure — should be much smaller than 1/N.
- For TRE deployments targeting DPUK governance standards, suggested starting values are **ε ≤ 1.0** and **δ ≤ 1/N².** [Citation Needed]

---

## Where DP Should Be Applied in deltab

### 1. Noise on exported model weights (output perturbation)

The safest and most practical DP mechanism for deltab is to add calibrated Gaussian noise to the model parameters before egress. The parameters eligible for egress are `b1`, `b2`, `b2sq`, `gamma`, `gammasq`, and the PCA components.

The **Gaussian mechanism** adds noise with standard deviation:

> σ = (sensitivity · √(2 ln(1.25/δ))) / ε

where **sensitivity** is the L2 sensitivity of the parameter with respect to a single subject's removal.

For the regression coefficient β₁:

> sensitivity(β₁) ≈ ‖**X_r⁺**‖₂ · max_age_range

In practice this requires bounding the feature norms. For IDP-based features normalised to zero mean and unit standard deviation, a conservative bound of 1.0 per feature is reasonable.

**Current deltab code — no DP implemented.** The required addition is:

```python
# Proposed addition to deltab.py — DP noise on model weights
import numpy as np

def _add_dp_noise(param, sensitivity, epsilon, delta):
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    return param + np.random.normal(0, sigma, param.shape)

# After training, before export:
# b1_private = _add_dp_noise(self.b1, sensitivity=0.1, epsilon=1.0, delta=1e-5)
```

### 2. Noise on PCA components

The PCA eigenvectors encode the covariance structure of the training data. Differentially private PCA (DP-PCA) adds noise to the sample covariance matrix before eigendecomposition:

> **C_priv** = **C** + **N** where N is a symmetric Gaussian noise matrix

This is more complex to implement and requires careful calibration of the noise level relative to the eigenvalue gap (the difference between the J-th and (J+1)-th eigenvalues). A small eigenvalue gap increases sensitivity.

Libraries such as **diffprivlib** (IBM) implement DP-PCA and can be used as a drop-in replacement for `sklearn.decomposition.PCA`.

```python
# Proposed replacement for PCA in deltab.py
# pip install diffprivlib
from diffprivlib.models import PCA as DPPCA

self.pca = DPPCA(n_components=ev_num, epsilon=1.0, data_norm=1.0)
self.x_reduced = self.pca.fit_transform(self.x_norm)
```

### 3. Noise on summary statistics before egress

If per-node summary statistics (mean delta, SD, quantiles) are approved for egress, the **Laplace mechanism** is appropriate for scalar statistics:

> output = true_statistic + Laplace(0, sensitivity/ε)

For the mean brain age delta across N subjects, sensitivity = max_delta_range / N. For a delta range of ±20 years and N = 500, sensitivity = 0.04. With ε = 1.0, the Laplace noise SD ≈ 0.04 years — negligible compared to the biological variation in delta.

---

## Privacy Budget Accounting

Each query or exported parameter consumes part of the privacy budget ε. When multiple parameters are released from the same training run, the **composition theorem** applies:

- **Sequential composition**: k releases each with budget εᵢ consume total budget Σεᵢ.
- **Advanced composition** (Rényi DP): tighter bounds for many releases.

For a single federated training run exporting β₁, β₂, β₂sq, and PCA components, the total budget consumption is roughly **4ε** under basic composition. With a target ε = 1.0 per release, the total budget is ε_total ≈ 4. This may be acceptable for a one-time model export but would not support repeated model updates.

For iterative federated learning (if implemented), use **Moments Accountant** or **Rényi DP** accounting, as implemented in TensorFlow Privacy or Opacus.

---

## Recommended DP Parameter Values for DPUK TRE Deployments

These are indicative starting values pending formal risk assessment by the DPUK data access committee:

| Parameter | Recommended value | Rationale |
|-----------|------------------|-----------|
| ε (per release) | ≤ 1.0 | Standard strong-privacy regime; consistent with published TRE DP guidance [Citation Needed] |
| δ | ≤ 1/N² | Below inverse-square of cohort size; ensures catastrophic failure probability is negligible |
| Noise mechanism | Gaussian | Suitable for continuous parameters with L2 sensitivity |
| DP-PCA | diffprivlib PCA | Drop-in replacement; maintains compatibility with sklearn pipeline |
| Minimum N for DP training | ≥ 100 | Below this, noise required for ε = 1.0 degrades model utility substantially |

---

## Limitations

DP gives probabilistic guarantees about the worst-case privacy loss. It does not guarantee that model outputs are uninformative — a sufficiently motivated adversary with access to many queries may still infer approximate membership for outlier subjects. DP should be treated as one layer of a defence-in-depth strategy, not as a complete solution.

For brain age applications specifically:

- Subjects with very unusual brain age deltas (large positive or negative values) remain at elevated risk even under DP, because their outlier status is itself informative.
- DP does not protect against disclosure of group-level information (e.g. that a particular diagnostic group has elevated brain age) — only individual-level membership.

---

## Running deltab Within a Teleport Federation

[Teleport](https://doi.org/10.5281/zenodo.10055358) is a DARE UK Phase 1 project that connects TREs across the UK's four nations — specifically the SAIL Databank / Secure eResearch Platform (SeRP) at Swansea University and the Scottish National Safe Haven (SNS) / EPCC at the University of Edinburgh — so that a researcher can access data held in multiple TREs from a single secure environment, without any data moving from its host site.

Teleport provides a **federated access layer**, not a federated learning framework. Data remains in place; what Teleport enables is coordinated job submission and result retrieval across connected TREs. For deltab this means:

| Step | What happens | DP requirement |
|------|-------------|----------------|
| Job submission | Researcher submits deltab training job to each TRE node via the Teleport environment | None — no data involved |
| Local training | deltab runs `train()` entirely within each TRE (SAIL or SNS) using locally held IDP data | No DP needed at this step |
| Weight extraction | Each node exports only `b1`, `b2`, PCA components (see [Egress & Privacy Risk Analysis](egress.md)) | DP noise added before export using `_add_dp_noise()` above |
| Egress via Teleport | Noisy weights pass through the TRE's standard SDC/SACRO egress process into the Teleport aggregation layer | TRE disclosure control review applies |
| Aggregation | Central aggregator (FedAvg) combines weights from SAIL and SNS nodes | Operates on already-noised outputs |
| Result return | Aggregated model returned to researcher in the Teleport environment | No raw subject data has moved |

### Key DP considerations for Teleport deployment

Because Teleport routes outputs through each TRE's existing egress gateway (SeRP at Swansea, eDRIS at SNS), the DP noise must be applied **before** egress — i.e. inside the TRE, as part of the deltab export step. The Teleport layer itself does not apply DP; it relies on each node's own disclosure controls.

The recommended approach for a two-node SAIL + SNS federation:

1. Each node trains deltab independently on its local cohort.
2. DP-noised weights are submitted for egress review under each TRE's SDC process.
3. Approved outputs flow through the Teleport aggregation layer to compute weighted-average parameters (FedAvg, weighted by N).
4. The aggregated model is returned to the researcher without any per-subject data leaving either TRE.

!!! note "Teleport and SeRP"
    SAIL Databank operates on SeRP (the Secure eResearch Platform at Swansea University), which is also the primary target TRE for DPUK studies. deltab's compute requirements (≤70 MB RAM, ≤10s runtime for 500 subjects) are well within SeRP's standard job allocation, making the Teleport pathway practical without requiring special compute resources.

---

## References

- Dwork, C. & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy. *Foundations and Trends in TCS*, 9(3–4). [https://doi.org/10.1561/0400000042](https://doi.org/10.1561/0400000042)
- Abadi, M. et al. (2016). Deep Learning with Differential Privacy. *ACM CCS*. [https://doi.org/10.1145/2976749.2978318](https://doi.org/10.1145/2976749.2978318)
- Holohan, N. et al. (2019). Diffprivlib: The IBM Differential Privacy Library. *arXiv*. [https://arxiv.org/abs/1907.02444](https://arxiv.org/abs/1907.02444)
- Kaissis, G. et al. (2021). End-to-end privacy preserving deep learning on multi-institutional medical imaging. *Nature Machine Intelligence*, 3, 473–484. [https://doi.org/10.1038/s42256-021-00337-8](https://doi.org/10.1038/s42256-021-00337-8)
- Orton, C. et al. (2023). TELEPORT: Connecting researchers to big data at light speed. DARE UK Phase 1 Final Report. Zenodo. [https://doi.org/10.5281/zenodo.10055358](https://doi.org/10.5281/zenodo.10055358)
