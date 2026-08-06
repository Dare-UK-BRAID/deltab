# Compute Requirements

This page provides estimated compute, memory, storage, and runtime requirements for running deltab at a federated TRE node with a cohort of **500 subjects** — a representative size for a DPUK SeRP site.

Estimates are based on typical BRAID pipeline IDP output dimensions (approximately 1,000 IDPs per subject from the structural, diffusion, and FLAIR pipelines combined) and the computational complexity of deltab's closed-form fitting algorithm.

---

## Dataset Dimensions (500 subjects)

| Quantity | Value | Notes |
|----------|-------|-------|
| N (subjects) | 500 | Representative DPUK SeRP site |
| F (IDP features) | ~1,000 | Full BRAID IDP set: T1 + diffusion TBSS + FLAIR WMH |
| J (PCA components retained) | ~100 | 10% of F, as recommended by Smith et al. |
| Age vector size | 500 × 1 | Float64 = 4 KB |
| Feature matrix size | 500 × 1,000 | Float64 = 4 MB |
| PCA-reduced matrix size | 500 × 100 | Float64 = 0.4 MB |

---

## Memory Requirements

All core computation in deltab uses NumPy in-process arrays. Peak memory usage occurs during PCA fitting on the full feature matrix.

| Operation | Peak RAM | Notes |
|-----------|---------|-------|
| Load feature matrix into memory | ~4 MB | 500 × 1,000 float64 |
| Normalise features (`_normalize`) | ~8 MB | Input + output arrays held simultaneously |
| PCA fit (`pca.fit_transform`) | ~50 MB | sklearn PCA uses full SVD by default; intermediate U, S, V matrices for 500 × 1,000 |
| Pseudoinverse (`np.linalg.pinv`) on X_r | ~1 MB | 100 × 500 matrix |
| Bias correction fitting | < 1 MB | Small matrices |
| `save()` pickle | ~8 MB | Full object including all training arrays |
| **Total peak** | **~70 MB** | Conservative estimate; well within standard TRE node memory |

!!! note "Memory is not a constraint"
    For 500 subjects and 1,000 IDPs, deltab requires well under 1 GB of RAM. Memory is not a practical constraint at any DPUK SeRP site.

---

## CPU Requirements

deltab is single-threaded by default. NumPy links to BLAS (OpenBLAS or MKL on most systems), which may internally parallelise matrix operations.

| Operation | Approximate time (1 core) | Notes |
|-----------|--------------------------|-------|
| Data loading | < 1 s | NumPy loadtxt on 4 MB file |
| Normalisation | < 1 s | Element-wise operations |
| PCA fit (500 × 1,000) | 1–3 s | SVD on moderately sized matrix |
| Pseudoinverse β₁ | < 1 s | pinv on 100 × 500 matrix |
| Bias correction β₂ | < 1 s | pinv on small matrices |
| Prediction (500 subjects) | < 1 s | Matrix multiplications |
| `save()` pickle | < 1 s | Object serialisation |
| **Total wall time** | **~5–10 s** | End-to-end, single core |

For cross-validation (k=10 folds as in the Smith et al. examples), wall time scales approximately linearly: **~50–100 s** total.

!!! note "No GPU required"
    deltab uses closed-form linear algebra, not iterative gradient descent. GPU acceleration provides no benefit. Standard TRE compute nodes (no GPU) are fully sufficient.

---

## Storage Requirements

| Data | Size | Classification |
|------|------|----------------|
| Input: age file | 4 KB | RESTRICTED |
| Input: IDP feature matrix | 4 MB | RESTRICTED |
| Input: confound matrix (optional) | < 1 MB | RESTRICTED |
| Output: delta vector (500 subjects) | 4 KB | RESTRICTED — SDC review required before egress |
| Model pickle (`save()`) | ~8 MB | RESTRICTED — must not be egressed |
| Model text output (`save_text()`) | ~20 MB (all arrays) | RESTRICTED — must not be egressed |
| Log files | < 1 MB | RESTRICTED |
| **Total local storage** | **~35 MB** | |

---

## Network Requirements

deltab itself has no runtime network requirements. In a federated deployment, the only network communication is between TRE nodes and the central aggregator for parameter exchange. This occurs outside deltab (mediated by the federated learning framework).

| Transfer | Size | Frequency |
|----------|------|-----------|
| β₁ (weights-only export) | ~0.8 KB (100 × 1 float64) | Once per training run |
| β₂, β₂sq | ~16 B (scalar / 2-vector) | Once per training run |
| PCA components (J × F) | ~0.8 MB (100 × 1,000 float64) | Once (if node-specific PCA used) |
| x_mean, x_std (normalisation) | ~16 KB (1,000 × 2 float64) | Once per training run |
| **Total per-node egress (weights only)** | **~1.6 MB** | Per training round |

For a 10-node federation with monthly model updates, total inter-node data transfer is approximately **16 MB/month** — negligible on any network.

---

## Recommended Minimum Node Specification

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores, 2.0 GHz | 4 cores, 3.0 GHz |
| RAM | 4 GB | 8 GB |
| Storage (secure enclave) | 1 GB | 10 GB (to accommodate IDP data from full pipeline) |
| OS | Linux (Ubuntu 20.04+) | Ubuntu 22.04 LTS |
| Python | 3.8 | 3.11 |
| Network (for parameter exchange) | 10 Mbps | 100 Mbps |
| GPU | Not required | Not required |

---

## Scaling Beyond 500 Subjects

If cohort size increases substantially (e.g. 5,000 or 50,000 subjects), the SVD step in PCA becomes the dominant cost:

| N | F | J | PCA time (1 core) | Peak RAM |
|---|---|---|------------------|---------|
| 500 | 1,000 | 100 | ~2 s | ~50 MB |
| 2,000 | 1,000 | 100 | ~10 s | ~200 MB |
| 5,000 | 1,000 | 100 | ~60 s | ~500 MB |
| 50,000 | 1,000 | 100 | ~30 min | ~5 GB |

For N > 5,000, use `sklearn.decomposition.TruncatedSVD` (randomised SVD) or `PCA(svd_solver='randomized')` to substantially reduce compute time at negligible accuracy cost. This change is not yet implemented in deltab.

---

## Summary

For the target deployment of 500 subjects at a DPUK SeRP node, deltab is computationally undemanding: a standard TRE compute node with 4 GB RAM, 2 CPU cores, and 1 GB of available secure storage is fully sufficient. The dominant practical constraint is not compute but data governance — specifically, the controls required to safely stage and review outputs before egress.
