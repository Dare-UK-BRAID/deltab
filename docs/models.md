# Prediction Models

deltab implements five prediction models. They differ in how they correct the age-dependence bias that affects naive brain age regression. All five are described in Smith et al. (2019).

---

## Model Overview

| Model | `--predict-model` | Bias correction | Quadratic aging | Use case |
|-------|-------------------|-----------------|-----------------|----------|
| Simple | `simple` | None | No | Comparison / replication only |
| Unbiased | `unbiased` | Linear correction | No | Linear age range, no quadratic effects |
| Unbiased Quadratic | `unbiased_quadratic` | Linear + quadratic | Yes | **Recommended default** |
| Alternate | `alternate` | Role reversal | No | Robustness check |
| Alternate Quadratic | `alternate_quadratic` | Role reversal | Yes | Robustness check with quadratic effects |

The Python API uses the `Model` class constants:

```python
from deltab import Model

Model.SIMPLE               # = 1
Model.UNBIASED             # = 2
Model.UNBIASED_QUADRATIC   # = 3  (default)
Model.ALTERNATE            # = 4
Model.ALTERNATE_QUADRATIC  # = 5
```

---

## Simple Model

**CLI:** `--predict-model simple`  
**API:** `Model.SIMPLE`

### What it does

The simple model fits a standard ridge-free ordinary least squares regression from the PCA-reduced feature matrix **X_r** to demeaned age **Ỹ**, and uses this fit to predict age for new subjects:

> **β₁** = **X_r⁺** · **Ỹ**  →  predicted age **Ŷ_B1** = **X_r** · **β₁**

Brain age delta is the residual:

> **δ₁** = **Ŷ_B1** − **Ỹ**

### Why it is biased

Because **β₁** is fitted to minimise the sum of squared residuals, the predicted ages cluster around the training mean. Subjects whose true age is far from the mean will have predictions pulled back towards the mean, creating apparent deltas that correlate strongly with true age. This is not a biological signal — it is a statistical artefact.

!!! warning "Do not use the simple model for real analyses"
    The simple model is included to reproduce the Smith et al. simulation results and to allow direct comparison with the corrected models. Using it in a real study will produce deltas that confound age effects with the biological signal of interest.

---

## Unbiased Model

**CLI:** `--predict-model unbiased`  
**API:** `Model.UNBIASED`

### What it does

The unbiased model takes the initial delta δ₁ from the simple model and removes its linear dependence on true age by regressing δ₁ onto **Ỹ** and subtracting the fitted component:

> **β₂** = **Ỹ⁺** · **δ₁**
>
> **δ₂** = **δ₁** − **Ỹ** · **β₂**

The result δ₂ is orthogonal to true age by construction — the correlation between predicted delta and true age is zero in the training data.

### When to use it

Use the unbiased model when:

- The relationship between brain features and age is expected to be approximately linear across the age range studied.
- The cohort covers a relatively narrow or uniform age range where quadratic effects are unlikely to be important.

---

## Unbiased Quadratic Model *(recommended)*

**CLI:** `--predict-model unbiased_quadratic`  
**API:** `Model.UNBIASED_QUADRATIC`

### What it does

This model extends the unbiased correction to also remove any quadratic dependence of δ₁ on true age. The correction is regressed onto the two-column age matrix **Y₂ = [Ỹ, Ỹ²ₒ]** where Ỹ²ₒ is the squared age term orthogonalised with respect to Ỹ (see [Theory](theory.md#step-5--quadratic-age-term)):

> **β₂q** = **Y₂⁺** · **δ₁**
>
> **δ₂q** = **δ₁** − **Y₂** · **β₂q**

The result is orthogonal to both Ỹ and Ỹ²ₒ, meaning it has neither a linear nor a quadratic relationship with true age.

### When to use it

This is the **recommended default** for most neuroimaging applications because:

- Brain ageing is rarely strictly linear. Rates of atrophy and white matter change accelerate in older decades.
- Cohorts spanning a wide age range (e.g. 45–80 years) are particularly susceptible to quadratic effects.
- The quadratic correction costs nothing computationally and rarely hurts when quadratic effects are absent.

!!! note "Default model"
    Both the CLI (`--predict-model unbiased_quadratic`) and the Python API (`model=Model.UNBIASED_QUADRATIC`) default to this model.

---

## Alternate Model

**CLI:** `--predict-model alternate`  
**API:** `Model.ALTERNATE`

### What it does

The alternate model takes a fundamentally different approach. Rather than regressing features onto age, it **reverses the direction** and regresses age onto features. The brain age delta is then defined as the component of the feature matrix that is not explained by age:

> **Γ** = **Ỹ⁺** · **X_r**
>
> **d** = **X_r** − **Ỹ** · **Γ**
>
> **δ_alt** = **d** · **Γ⁺**

Conceptually, this asks: "given the subject's features, what is left over after removing the part that can be explained by knowing their true age?" This residual maps back to a scalar delta via the pseudoinverse of Γ.

### When to use it

The alternate model provides a robustness check. If results are consistent between the unbiased and alternate models, this increases confidence that the observed delta reflects a genuine biological signal rather than a statistical property of the modelling approach.

---

## Alternate Quadratic Model

**CLI:** `--predict-model alternate_quadratic`  
**API:** `Model.ALTERNATE_QUADRATIC`

### What it does

The alternate quadratic model extends the alternate model by using the two-column age matrix **Y₂ = [Ỹ, Ỹ²ₒ]** instead of **Ỹ** alone, removing both linear and quadratic age effects from the feature matrix before computing the delta:

> **Γ_q** = **Y₂⁺** · **X_r**
>
> **d** = **X_r** − **Y₂** · **Γ_q**
>
> **δ_alt-q** = **d** · **Γ_q[0,:]⁺**

### When to use it

Use alongside the `unbiased_quadratic` model as a secondary confirmation when quadratic brain ageing effects are expected.

---

## Choosing a Model

For most research applications the decision tree is straightforward:

```
Is this for replication / method comparison?
  YES → simple (but always also report a corrected model)
  NO  → Does your cohort span >20 years or do you expect non-linear ageing?
          YES → unbiased_quadratic (default)
          NO  → unbiased
        Do you want a robustness check?
          YES → also run alternate or alternate_quadratic
```

Smith et al. (2019) Table 1 shows that the corrected models substantially reduce mean absolute delta error and increase correlation with the true delta across all simulation conditions, while the simple model degrades sharply as the number of PCA components increases.

---

## Cross-Validation

When training and testing on the same dataset, results should always be obtained via **cross-validation** to avoid overfitting. The `smith_results_table1.py` and `smith_results_table2.py` examples implement a k-fold cross-validation approach:

1. Subjects are randomly assigned to one of k groups.
2. In each fold, one group is held out, the model is trained on the remaining subjects, and predictions are generated for the held-out group.
3. Predictions are assembled across all folds to give a complete out-of-sample set of deltas.

See [Examples — Cross-validation](examples.md#cross-validation) for a worked implementation.

---

## References

- Smith, S.M. et al. (2019). Estimation of brain age delta from brain imaging. *NeuroImage*, 200, 528–539. [https://doi.org/10.1016/j.neuroimage.2019.06.017](https://doi.org/10.1016/j.neuroimage.2019.06.017)
