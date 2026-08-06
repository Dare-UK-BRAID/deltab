# Secure Aggregation & MPC

Secure aggregation and Multi-Party Computation (MPC) allow a central aggregator to compute functions over model parameters from multiple TRE nodes without learning the individual node's contributions. For federated brain age estimation, these techniques prevent the central server from inferring which node contributed anomalous parameters — which could in turn reveal something about that node's cohort.

---

## Why Secure Aggregation Matters for deltab

In a naive federated deployment, each TRE node sends its model weights (β₁, β₂, PCA components, normalisation statistics) directly to the central aggregator. Even if each node applies differential privacy noise before egress, the central server can still:

- Compare parameters across nodes to identify which node has an unusual cohort
- Combine parameters from multiple nodes to partially reconstruct the training data distribution
- Perform node-level membership inference (attributing a model anomaly to a specific node, and by extension to that node's subjects)

Secure aggregation prevents the central server from seeing individual node contributions: it receives only the **sum** (or **weighted average**) of parameters across all participating nodes.

---

## Aggregation Strategies for deltab Parameters

### Federated averaging of model weights

The standard approach (McMahan et al., 2017 — FedAvg) averages the model weights from each node weighted by cohort size:

> β₁_global = Σ (Nᵢ / N_total) · β₁ᵢ

For deltab this is straightforward for `b1`, `b2`, `b2sq` (all scalar or low-dimensional vectors). The challenge is aggregating PCA components, which are not straightforwardly averageable because each node's PCA may select different axes.

**PCA aggregation options:**

| Approach | Description | Suitability |
|----------|-------------|-------------|
| Federated PCA (power iteration) | Iteratively refine shared eigenvectors across nodes | Best accuracy; requires multiple communication rounds |
| Concatenate and re-decompose | Each node sends its top-J components; central server decomposes the concatenated matrix | Simple; leaks node-specific components unless encrypted |
| Use a shared pre-trained PCA | Apply a single PCA fitted on a normative public dataset (e.g. UK Biobank summary statistics) across all nodes | Removes within-federation PCA sharing entirely; recommended for DPUK |

The shared normative PCA approach is the most privacy-preserving: the PCA is fitted once on publicly released data and distributed to all nodes, so no node-specific covariance information is transmitted.

### Normalisation statistics aggregation

The feature means (`x_mean`) and standard deviations (`x_std`) must be harmonised across nodes before model averaging is meaningful. Two approaches:

- **Global normalisation**: Compute global mean and SD using secure aggregation (Σᵢ Nᵢ·meanᵢ / N_total) before training begins. Each node normalises its data using the global statistics.
- **Local normalisation + re-weighting**: Each node normalises locally; aggregated weights are corrected analytically. More complex but avoids a pre-training communication round.

---

## Secure Aggregation Protocols

### Pairwise masking (Bonawitz et al., 2017)

Each pair of nodes agrees on a random mask. Node A adds the mask to its parameters; node B subtracts it. The central server sees masked values from each node; the masks cancel in the sum, revealing only the aggregate.

This protocol is robust to node dropout and does not require a trusted third party. It has been standardised in several federated learning frameworks (e.g. PySyft, FATE, Flower).

**Applicability to deltab:** Directly applicable. Each node masks β₁ᵢ before transmission; the aggregator computes Σ masked β₁ᵢ = Σ β₁ᵢ (masks cancel). The aggregator learns only the sum, not individual contributions.

### Homomorphic Encryption (HE)

Nodes encrypt their parameters under a public key. The central server aggregates encrypted ciphertexts and decrypts only the final aggregate using the private key. No plaintext node parameters are ever exposed to the server.

**Applicability to deltab:** Feasible for `b1`, `b2`, `b2sq` (low-dimensional). High computational overhead for PCA components (J × F matrix). Partially homomorphic encryption (Paillier scheme) is sufficient for summation-only aggregation.

### Secure Multi-Party Computation (SMPC)

A generalisation of pairwise masking where a threshold of nodes must collude before any individual contribution is revealed. SMPC with a (t, n)-threshold scheme requires t+1 nodes to cooperate to reconstruct any individual share.

**Applicability to deltab:** Suitable for small federations (3–10 nodes) where each node is a separate institution with adversarial interests. For DPUK, a (2, N)-threshold (any 2 nodes) is a reasonable starting point.

---

## Gradient Sharing Considerations

deltab uses closed-form ordinary least squares (pseudoinverse) rather than iterative gradient descent. This means there are no gradients to share — only the final parameter vectors. This is advantageous from a security perspective because:

- Gradient inversion attacks (Zhu et al., 2019) are not directly applicable
- There is no per-iteration communication round that could leak intermediate states
- The attack surface is limited to the final exported parameters

However, if deltab is extended to use iterative fitting (e.g. stochastic gradient descent for very large feature sets), gradient sharing risks would need to be re-evaluated.

---

## Recommended Architecture for DPUK

For a DPUK SeRP federated deployment, the recommended architecture is:

1. **Pre-training**: Distribute a shared normative PCA (fitted on public UK Biobank summary statistics) to all nodes. No node-specific PCA is computed.

2. **Local training**: Each node runs `train()` using the shared PCA (replacing `pca.fit_transform()` with `pca.transform()` using the pre-fitted components). Only β₁, β₂, β₂sq are fitted locally.

3. **Secure aggregation**: Each node applies DP noise to β₁, β₂, β₂sq, then transmits masked (pairwise-masked or HE-encrypted) parameters to the central server.

4. **Central aggregation**: The server decrypts and computes the weighted average of β₁, β₂, β₂sq across nodes.

5. **Global model distribution**: The averaged parameters are broadcast back to all nodes for federated inference.

6. **Output egress**: Per-node inference results (delta vectors) remain local. Only aggregated group-level statistics (with DP noise and N ≥ threshold) are approved for egress.

---

## Frameworks and Libraries

| Framework | Notes |
|-----------|-------|
| **Flower (flwr)** | Lightweight federated learning framework; easy integration with sklearn-style models; supports custom aggregation strategies |
| **PySyft** | SMPC and HE support; more complex; suited to high-security deployments |
| **FATE** | Production-grade FL platform; native HE and SMPC; designed for institutional federation |
| **diffprivlib** | IBM DP library; provides DP-PCA and DP linear regression compatible with sklearn API |
| **TF Privacy / Opacus** | DP-SGD for deep learning; not directly applicable to deltab's closed-form fitting unless reframed as SGD |

---

## References

- Bonawitz, K. et al. (2017). Practical Secure Aggregation for Privacy-Preserving Machine Learning. *ACM CCS*. [https://doi.org/10.1145/3133956.3133982](https://doi.org/10.1145/3133956.3133982)
- McMahan, B. et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. *AISTATS*. [https://arxiv.org/abs/1602.05629](https://arxiv.org/abs/1602.05629)
- Zhu, L. et al. (2019). Deep Leakage from Gradients. *NeurIPS*. [https://arxiv.org/abs/1906.08935](https://arxiv.org/abs/1906.08935)
- Rieke, N. et al. (2020). The future of digital health with federated learning. *npj Digital Medicine*, 3, 119. [https://doi.org/10.1038/s41746-020-00323-1](https://doi.org/10.1038/s41746-020-00323-1)
