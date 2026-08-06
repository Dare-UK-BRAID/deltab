# deltab — Brain Age Estimation

**deltab** is a Python package for estimating *brain age* and *brain age delta* from neuroimaging-derived phenotypes (IDPs) or other brain measurements. It implements the modelling framework described in:

> Smith, S.M. et al. (2019). Estimation of brain age delta from brain imaging. *NeuroImage*, 200, 528–539. [https://doi.org/10.1016/j.neuroimage.2019.06.017](https://doi.org/10.1016/j.neuroimage.2019.06.017)

---

## What is Brain Age?

The brain ages throughout the lifespan in ways that are detectable from structural and functional neuroimaging data. Features such as cortical thickness, white matter integrity, and subcortical volumes all change systematically with age and can be used to train a regression model that predicts a person's age from their brain scan alone.

The **predicted age** produced by such a model is sometimes called *brain age*. In a healthy individual, brain age should closely match chronological age. In someone with accelerated neurodegeneration or neurodevelopmental differences, brain age may be older than their actual age.

## What is Brain Age Delta?

**Brain age delta** (δ) is the difference between a person's predicted brain age and their true chronological age:

**δ = brain age − true age**

A positive delta indicates the brain appears older than expected; a negative delta indicates the brain appears younger. Brain age delta has been proposed as a biomarker for brain health, with positive deltas associated with neurodegenerative disease, poor cardiovascular health, and cognitive decline.

## The Bias Problem

A naïve linear regression approach to brain age prediction produces a result that is systematically **biased**: younger participants tend to have positive deltas (brain appears older) and older participants tend to have negative deltas (brain appears younger). This arises because regression towards the mean is an intrinsic property of ordinary least squares fitting. The bias is present even when the underlying model is correct, and it invalidates direct comparisons of delta across cohorts with different age distributions.

Smith et al. (2019) identify and characterise several sources of this bias and derive corrected estimators that remove it. **deltab** implements all of these corrected models alongside the original naive model for comparison.

!!! warning "Always use a corrected model"
    The `simple` model is provided for comparison and to reproduce the Smith et al. simulation results. For any real study, use `unbiased_quadratic` (the default) or one of the other corrected models.

---

## What deltab Does

deltab takes two inputs:

- A vector of **true chronological ages** for a set of subjects (one value per subject)
- A matrix of **brain imaging features** for the same subjects (one row per subject, one column per feature — typically IDPs from a pipeline such as BRAID)

It trains a linear model linking the features to age, applies PCA reduction to handle high-dimensional feature spaces, and then produces a corrected brain age delta for each subject using one of five prediction models.

The trained model can be saved and reloaded for use with new subjects, supporting the common workflow where a normative model is trained on a large healthy control dataset and then applied to a clinical cohort.

---

## Quick Reference

| Topic | Where to look |
|-------|--------------|
| Why corrected models are needed | [Theory — The Bias Problem](theory.md#the-bias-problem) |
| All five prediction models explained | [Prediction Models](models.md) |
| Installing the package | [Installation](installation.md) |
| Running from the command line | [Command Line Interface](cli.md) |
| Using the Python API | [Python API](api.md) |
| Reproducing Smith et al. simulation results | [Examples](examples.md) |

---

## Quick Start

Install and run a basic brain age prediction in three steps:

```bash
pip install deltab
```

```bash
deltab \
  --train-ages TRUE_AGES.txt \
  --train-features IDPS.txt \
  --predict-model unbiased_quadratic \
  --predict delta \
  --predict-output BRAIN_AGE_DELTA.txt
```

Or from Python:

```python
from deltab import BrainDelta, Model
import numpy as np

ages = np.loadtxt("TRUE_AGES.txt")
features = np.loadtxt("IDPS.txt")

b = BrainDelta()
b.train(ages, features, ev_proportion=0.1)  # retain top 10% of PCA components
delta = b.predict(ages, features, model=Model.UNBIASED_QUADRATIC, return_delta=True)
np.savetxt("BRAIN_AGE_DELTA.txt", delta)
```

---

## References

- Smith, S.M. et al. (2019). Estimation of brain age delta from brain imaging. *NeuroImage*, 200, 528–539. [https://doi.org/10.1016/j.neuroimage.2019.06.017](https://doi.org/10.1016/j.neuroimage.2019.06.027)
- Cole, J.H. et al. (2019). Multimodality neuroimaging brain-age in UK Biobank: relationship to biomedical, lifestyle, and cognitive factors. *Neurobiology of Aging*, 92, 34–43. [https://doi.org/10.1016/j.neurobiolaging.2020.03.014](https://doi.org/10.1016/j.neurobiolaging.2020.03.014)
- Franke, K. & Gaser, C. (2019). Ten years of BrainAGE as a neuroimaging biomarker of brain aging: what insights have we gained? *Frontiers in Neurology*, 10, 789. [https://doi.org/10.3389/fneur.2019.00789](https://doi.org/10.3389/fneur.2019.00789)
