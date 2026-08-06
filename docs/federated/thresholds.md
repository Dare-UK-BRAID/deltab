# Disclosure Thresholds

Disclosure thresholds define the minimum conditions under which a statistical output may be approved for egress from a TRE node. For federated brain age estimation using deltab, thresholds apply both to model parameters and to derived summary statistics.

The thresholds described here are informed by DPUK data governance requirements, the ONS Five Safes framework, and established practice in TRE-based neuroimaging research. Where specific DPUK thresholds differ from general guidance, DPUK requirements take precedence.

---

## General Principles

The core principle is **minimum cell size**: no output derived from fewer than a specified number of subjects should be released without enhanced scrutiny. For neuroimaging TREs, the following hierarchy applies:

1. **Primary threshold** — the absolute minimum N for any output to be considered for egress.
2. **Secondary threshold** — the minimum N for subgroup analyses (diagnostic group, sex, age band, scanner site).
3. **Dominance rule** — no single subject should contribute disproportionately to an aggregate statistic (typically: no individual accounts for > 10% of any cell).
4. **Complementary suppression** — if one cell is suppressed because it falls below threshold, any cell whose value could be derived from it by subtraction must also be suppressed.

---

## Recommended Thresholds for deltab Outputs

### Model parameters

| Output | Minimum N | Additional conditions |
|--------|-----------|----------------------|
| β₁ (regression coefficients) | ≥ 100 | DP noise applied; no reconstruction possible from N and J alone |
| β₂, β₂sq (bias correction) | ≥ 100 | Scalar or 2-vector; low re-identification risk |
| PCA components | ≥ 100 | Prefer shared normative PCA; node-specific PCA suppressed for N < 200 |
| x_mean, x_std (normalisation stats) | ≥ 50 | Population-level statistics; low individual risk |
| y_mean (training age mean) | ≥ 20 | Risk increases if combined with known age ranges |
| N (number of subjects) | ≥ 20 | Below this, exact N may be disclosive |

### Summary statistics from prediction

| Output | Minimum N | Rounding/suppression |
|--------|-----------|---------------------|
| Mean brain age delta (whole cohort) | ≥ 20 | Round to 1 decimal place |
| SD of brain age delta | ≥ 20 | Round to 1 decimal place |
| Mean delta by diagnostic group | ≥ 20 per group | Suppress groups below threshold; do not publish complementary group |
| Mean delta by age band | ≥ 20 per band | 5-year age bands minimum width |
| Mean delta by sex | ≥ 20 per sex | Standard; rarely a risk in large cohorts |
| Mean delta by scanner site | ≥ 20 per site | Site identity may be sensitive in small federation |
| Full per-subject delta vector | Prohibited | Never egress without individual DAC approval |
| Quantiles / percentiles | ≥ 100 | Extreme quantiles (< 5th, > 95th) suppressed for N < 200 |

---

## Age-Specific Considerations

Age is a quasi-identifier. In neuroimaging cohorts with narrow age ranges (e.g. 65–75 years), fine-grained age data in combination with imaging features can uniquely identify individuals. Specific controls:

- **y_mean** should be rounded to the nearest year before egress if N < 100.
- **Age band width** for subgroup analyses should be at minimum 5 years; preferably 10 years for small sites (N < 200).
- **True ages** (`ytrain`) must never be egressed. Even demeaned or normalised ages (`y_demean`) are restorable given `y_mean` and must not be transmitted.

---

## IDP-Specific Considerations

Imaging-derived phenotypes are not direct identifiers, but in combination they can be highly distinctive. Research has shown that resting-state connectivity profiles can identify individuals with >90% accuracy across sessions, and that structural IDPs (cortical thickness, subcortical volumes) have comparable discriminatory power. [Citation Needed]

Specific controls for IDP-derived outputs:

- **Per-subject IDP vectors** (`xtrain`, `x_norm`, `x_reduced`) must never be egressed.
- **Feature-level summary statistics** (mean and SD per IDP across cohort) are low risk for N ≥ 50 but should be checked for extreme values that may reflect a single subject.
- **PCA components** (eigenvectors of the IDP covariance matrix) should be treated as MEDIUM risk — they encode distributional properties of the cohort's IDP space.

---

## Small Cell Suppression Rules

When reporting aggregated brain age delta statistics by subgroup:

1. Any subgroup with N < 20 is suppressed (replaced with `<20` or `*`).
2. If suppression of one group makes a complementary group implicitly calculable (e.g. total minus suppressed = revealed), the complementary group is also suppressed.
3. When more than 30% of cells in a table are suppressed, the entire table is withheld.
4. Summaries must not allow back-calculation of suppressed cells via marginal totals.

---

## Output Rounding Protocol

All egressed statistics should be rounded before leaving the node:

| Statistic type | Rounding |
|---------------|----------|
| Mean delta (years) | 1 decimal place |
| SD (years) | 1 decimal place |
| N (number of subjects) | Round to nearest 5 for N < 100; report exact for N ≥ 100 |
| Regression coefficients | 3 significant figures |
| Correlation coefficients | 2 decimal places |
| p-values | 3 decimal places; report as < 0.001 below threshold |

---

## SACRO Integration

The TREvolution SACRO tool (Statistical Analysis for Controlled Release of Outputs) provides automated checks for many of these thresholds. For deltab outputs, the following SACRO checks are relevant:

- **Frequency check**: Verifies minimum cell size before any tabular output is released.
- **Dominance check**: Flags cells where a small number of subjects contribute the majority of the statistic.
- **Regression output check**: Examines regression coefficient tables for outputs that could enable reconstruction.

SACRO does not currently have a native check for machine learning model weights. A custom SACRO plugin implementing the checks described in this page should be developed as part of the TREvolution/DPUK integration.

---

## References

- UK Data Service (2021). Safe Researcher Training — Handling Sensitive Data. [https://www.ukdataservice.ac.uk](https://www.ukdataservice.ac.uk)
- Office for National Statistics (2023). Microdata Output Guidance. [https://www.ons.gov.uk](https://www.ons.gov.uk)
- DPUK Data Access and Governance Policy. Dementias Platform UK. [https://www.dementiasplatform.uk](https://www.dementiasplatform.uk)
- TREvolution SACRO. AI-SDC GitHub. [https://github.com/AI-SDC/SACRO](https://github.com/AI-SDC/SACRO)
- Sudlow, C. et al. (2015). UK Biobank: An Open Access Resource for Identifying the Causes of a Wide Range of Complex Diseases of Middle and Old Age. *PLOS Medicine*, 12(3). [https://doi.org/10.1371/journal.pmed.1001779](https://doi.org/10.1371/journal.pmed.1001779)
