# MIMIC-IV v2.2 synthetic cohort of 100 patients

This versioned project creates a reproducible, privacy-conscious synthetic MIMIC-IV educational cohort with 100 unique patients, 73 predictors, hospital length-of-stay regression, and 30-day readmission classification.

## Existing work reused

The earlier work in the parent `MIMIC` directory (now `data/profiles_1000/` at the repo root) contained a prior 1,000-profile wide dataset, task-specific CSVs, privacy summaries, model results, and notebooks. Those files are preserved. This project reuses the verified MIMIC-IV v2.2 installation, the earlier privacy-aware distribution-learning approach, and its cached identifier-protected source modeling table. It does not overwrite parent files.

## Source schemas and source-like synthetic tables

When regenerating, the pipeline recursively resolves the folder in the `MIMIC_IV_ROOT` environment variable and verifies actual headers before generation. Synthetic source-like CSVs preserve exact source columns for:

- `patients`
- `admissions`
- `icustays`
- `diagnoses_icd`
- `procedures_icd`
- `services`
- `labevents`
- `chartevents`
- `inputevents` (used for verified vasopressor indicators)

No table is empty. Synthetic identifiers are newly generated and do not reproduce MIMIC identifiers. The source-like tables retain exact long-format names; the wide modeling table uses documented derived names mapped in `outputs/tables/feature_dictionary.csv`.

## Cohort and generation

- Fixed seed: 42.
- Exactly 100 unique synthetic subjects.
- One index admission per subject in the modeling table.
- Twenty synthetic subjects receive a subsequent urgent admission 2–28 days after index discharge to demonstrate readmission.
- ICU stays are generated for a clinically plausible subset.
- A cached eligible MIMIC source cohort is sampled only to learn broad admission-level distributions; continuous values are perturbed and complete source rows and identifiers are never exported.
- Vitals, severity, disease burden, labs, LOS, ICU utilization, and readmission are generated jointly through a correlated latent severity structure.

This is not a formal privacy guarantee. `privacy_similarity_report.csv` documents exact common-field checks.

## 73 predictors

The approved configuration contains:

- 8 demographics/admission predictors;
- 19 first-24-hour vital summaries;
- 14 comorbidity features, including `comorbidity_count`;
- 19 first-24-hour laboratory/chart features;
- 13 service/intervention features.

Verified item IDs and official dictionary labels come from `d_labitems` and `d_items`. ICD-10 code mappings are explicit in the project and represent a compact educational mapping, not a complete validated Charlson/Elixhauser implementation. The schema audit excludes anything that cannot be verified.

## Outcomes and timing

`hospital_los_days = (dischtime - admittime) / 24 hours`. LOS predictors are restricted to information available at admission or in the first 24 hours. Discharge time, discharge location, ICU outtime/LOS, future procedures/services, and readmission are excluded.

`readmitted_30d = 1` when a later synthetic urgent inpatient admission starts more than 0 and no more than 30 days after index discharge. Same-day records are excluded. Generated readmissions are unplanned/urgent; transfers are represented within services and are not counted as admissions. Index deaths are excluded by construction. Observation stays are not generated as readmissions. Subsequent-admission variables never enter predictors.

## EDA and models

EDA includes cohort, demographic, comorbidity, clinical, missingness, co-occurrence, and domain summaries plus high-resolution figures.

Models use scikit-learn `Pipeline` and `ColumnTransformer` preprocessing with median/mode imputation, one-hot encoding, and scaling. Repeated 5-fold cross-validation (three repeats) is used because 100 patients are insufficient for strong conclusions. LOS models compare dummy mean, Elastic Net, random forest, and histogram gradient boosting. Readmission models compare dummy prior, class-weighted logistic regression, random forest, and histogram gradient boosting. Permutation importance provides top predictors.

## Run

```bash
# Regeneration needs credentialed MIMIC-IV; set MIMIC_IV_ROOT first (see top-level README)
cd synthetic_cohort_100
python run_pipeline.py
pytest -q
```

Python 3.11 is supported. All paths use `pathlib`.

## Limitations

This cohort is small and synthetic. Generated prevalence and associations are workshop-oriented and do not establish clinical utility, calibration, fairness, external validity, or causality. Source-code mappings are deliberately compact. Synthetic data and models are not clinically validated and must never be used for clinical decisions.

