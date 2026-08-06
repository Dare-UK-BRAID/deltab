# Federated Brain Age Estimation Requirements

!!! note "Derived Work"
    This section adapts the brain age estimation framework of Smith et al. (2019) for deployment within Trusted Research Environments (TREs) under federated analytics (FA) frameworks, specifically targeting DPUK SeRP nodes and the TREvolution/SACRO governance model.

!!! info "Target Audience"
    TRE operators, data access committees, and neuroimaging researchers deploying deltab across federated DPUK nodes.

---

## Overview

Brain age estimation using **deltab** in a federated context presents unique privacy and governance challenges. Unlike a centralised analysis in which all subject data resides on a single server, a federated deployment trains or applies the model separately within each TRE node and shares only derived quantities — model weights, summary statistics, or aggregated deltas — with a coordinating central server.

This section documents the requirements, risks, and controls needed to deploy deltab safely within a federated neuroimaging infrastructure such as DPUK SeRP, following the disclosure risk principles of the TREvolution SACRO project.

---

## Architecture Overview

A federated deltab deployment consists of:

- **N TRE nodes** (e.g. DPUK SeRP sites) — each holds a cohort of subjects whose raw imaging data and ages never leave the node.
- **A central aggregator** — receives only approved, disclosure-controlled outputs from each node.
- **A disclosure control layer** — implemented via SACRO or an equivalent Statistical Disclosure Control (SDC) mechanism sitting between each node's compute environment and its egress gateway.

```
┌─────────────────┐     approved outputs only     ┌──────────────────┐
│  TRE Node A     │ ─────────────────────────────▶ │                  │
│  (DPUK SeRP)    │                                 │  Central         │
│  deltab train() │                                 │  Aggregator      │
└─────────────────┘                                 │                  │
┌─────────────────┐     approved outputs only     │                  │
│  TRE Node B     │ ─────────────────────────────▶ │                  │
│  (DPUK SeRP)    │                                 └──────────────────┘
│  deltab train() │
└─────────────────┘
```

The central aggregator **never receives** raw ages, raw IDPs, or per-subject predictions unless those outputs have been reviewed and approved by the TRE data access committee and passed through the disclosure control gateway.

---

## Federated Deployment Modes

Three deployment modes are applicable to deltab:

| Mode | What happens at each node | What is shared centrally |
|------|--------------------------|--------------------------|
| **Federated training** | Full `train()` call on local cohort | Model weights (β₁, β₂, PCA components, normalisation statistics) after SDC review |
| **Federated inference** | Local `train()` + `predict()` | Aggregated delta statistics (mean, SD, quantiles) per pre-agreed subgroup |
| **Transfer learning** | Central model loaded via `load()`, local `predict()` only | Per-subject delta vectors after SDC review |

In all modes, raw subject data (**ages, IDPs, per-subject deltas**) must remain within the TRE node unless explicitly approved for egress by the data access committee.

---

## Subpages

| Topic | Page |
|-------|------|
| What the code transmits and what risks this creates | [Egress & Privacy Risk Analysis](egress.md) |
| How membership attacks apply to brain age models | [Membership Inference Attacks](membership-attacks.md) |
| Applying differential privacy to deltab outputs | [Differential Privacy](differential-privacy.md) |
| Secure aggregation and multi-party computation | [Secure Aggregation & MPC](secure-aggregation.md) |
| Minimum cell sizes and suppression thresholds | [Disclosure Thresholds](thresholds.md) |
| What data and software each federated node needs | [Node Data & Software Requirements](node-requirements.md) |
| Compute sizing for 500 subjects | [Compute Requirements](compute-requirements.md) |
