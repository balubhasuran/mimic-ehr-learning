# Learning EHR Data Processing & Modeling with a Synthetic MIMIC-IV Cohort

Hands-on notebooks for **clinicians and clinical informatics professionals** who are new to working with electronic health record (EHR) data. You will learn how MIMIC-IV-style tables relate, how to build a leakage-safe analysis dataset, and how to train and *critically evaluate* simple prediction models — all on **synthetic data you can use immediately, with no credentialed access**.

> ⚠️ **Educational use only.** All data here are synthetic. Models and results are not clinically validated and must never inform patient care. See [docs/PRIVACY_AND_DATA_USE.md](docs/PRIVACY_AND_DATA_USE.md).

## What you will learn

| Topic | Where |
|---|---|
| How EHR tables (`patients`, `admissions`, `icustays`, `labevents`, …) link via `subject_id` / `hadm_id` / `stay_id` | Notebooks 01, 00 |
| Defining a cohort, row grain, index time, and observation window | 01, 00 |
| Safe joins: aggregate one-to-many tables *before* merging | 01, 00 |
| Data-quality checks; missingness in EHR (not ordered ≠ bad data) | 01 |
| Outcome definitions: length of stay (LOS), 30-day readmission, in-hospital mortality | 01, 02, 03 |
| Data leakage: what you may and may not use as a predictor | 01, 02, `config/leakage_exclusions.yaml` |
| Preprocessing mixed clinical data (imputation, encoding, scaling) with scikit-learn pipelines | 01–03 |
| Model comparison vs. a dummy baseline; cross-validation on small samples | 01–03 |
| Metrics for imbalanced outcomes (AUROC, AUPRC, calibration, sensitivity/specificity) | 01, 02 |
| Interpretability: coefficients, permutation importance, SHAP | 01, 02, 03 |
| Subgroup (sex) performance and thinking about bias | 01 |

## Repository layout

```
mimic-ehr-learning/
├── notebooks/
│   ├── 01_beginner_workshop_synthetic_cohort_100.ipynb   ← START HERE
│   ├── 02_mortality_binary_classification.ipynb
│   ├── 03_length_of_stay_regression.ipynb
│   └── 00_source_data_processing_pipeline.ipynb          ← reference; needs real MIMIC-IV
├── synthetic_cohort_100/          # 100-patient project: source-like tables + modeling table + code
│   ├── data/synthetic_tables/     # 9 MIMIC-IV-shaped tables (patients, admissions, labevents, ...)
│   ├── data/synthetic_mimic_100_modeling.csv   # 1 row per patient, 73 predictors + 2 outcomes
│   ├── config/                    # feature list, leakage exclusions, plausibility ranges
│   ├── src/  run_pipeline.py  tests/           # generator, EDA, models, tests
│   └── outputs/                   # figures, tables, reports from a completed run
├── data/profiles_1000/            # 1,000 synthetic admission-level profiles + task tables
└── docs/                          # learning path and privacy notes
```

## Quick start

Requires Python 3.10+ (developed on 3.11).

```bash
git clone https://github.com/<your-username>/mimic-ehr-learning.git
cd mimic-ehr-learning
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebooks/
```

Open notebook **01** and run it top to bottom. Notebooks locate the data relative to the repo, so Jupyter can be started from the repo root or from `notebooks/`.

Run the tests (schema tests that need real MIMIC-IV are skipped automatically):

```bash
cd synthetic_cohort_100 && python -m pytest -q
```

## The datasets

### A. 100-patient cohort — `synthetic_cohort_100/`
Small enough to inspect by hand; mirrors real MIMIC-IV table structure with exact column names.
- **9 source-like tables** in `data/synthetic_tables/`.
- **Modeling table**: 100 patients, 73 predictors (demographics, first-24-h vitals, comorbidities, first-24-h labs, services/interventions).
- **Tasks**: hospital LOS regression; 30-day readmission classification (20 positives / 80 negatives).
- Trace every feature to its source table/column in `outputs/tables/feature_dictionary.csv`.
- With n = 100 results are deliberately noisy — a good lesson in why a dummy baseline can win.

### B. 1,000-profile set — `data/profiles_1000/`
Wide admission-level table (42 columns) with three ready-made tasks and fixed train/validation/test splits (`split_assignments.csv`):

| Task | File | Target | Note |
|---|---|---|---|
| Disease class (11 classes) | `disease_classification_dataset.csv` | `primary_disease_class` | Hard; baseline macro-F1 ≈ 0.2–0.3 |
| Mortality | `mortality_prediction_dataset.csv` | `in_hospital_mortality` | Only ~2% positive (22 events) — an imbalance lesson |
| LOS | `length_of_stay_regression_dataset.csv` | `length_of_stay_days` | Right-skewed; log transform helps |

Supporting files: `data_dictionary.csv` (per-variable definitions and leakage risk), `leakage_audit.csv`, `baseline_results.csv`, `confusion_matrices.csv`, `generation_quality_report.csv`, `privacy_similarity_report.csv`, `cohort_flow.csv`, `source_vs_synthetic_summary.csv`.

## Suggested learning path

See [docs/LEARNING_PATH.md](docs/LEARNING_PATH.md) for a ~1-day plan with exercises:
1. **Notebook 01** — tables, joins, cohort, leakage, EDA, LOS + readmission models, calibration, SHAP.
2. **Notebook 02** — mortality classification with validation-based model selection.
3. **Notebook 03** — LOS regression with four models and SHAP.
4. **Notebook 00** — how the real-data source table is built (read-only unless you hold MIMIC-IV access).

## How the synthetic data were made (short version)

Distributions were learned from an eligible MIMIC-IV v2.2 adult cohort; new records were generated with perturbed continuous values and newly created identifiers. No real rows or identifiers are included, and candidate profiles too similar to a real holdout were rejected. Details and limitations: [docs/PRIVACY_AND_DATA_USE.md](docs/PRIVACY_AND_DATA_USE.md).

## Regenerating the data (optional)

`synthetic_cohort_100/run_pipeline.py` regenerates the 100-patient project. It requires a **credentialed** local copy of MIMIC-IV v2.2:

```bash
export MIMIC_IV_ROOT=/path/to/mimic-iv-2.2                          # PowerShell: $env:MIMIC_IV_ROOT="..."
export MIMIC_SOURCE_CACHE=/path/to/source_modeling_table.parquet    # optional
cd synthetic_cohort_100 && python run_pipeline.py
```

Learners do **not** need this — all outputs are already included.

## Limitations

- Synthetic data reproduce broad marginal structure, not true clinical realism or causal relationships.
- Small samples (100 / 1,000) and rare outcomes make performance estimates unstable.
- Comorbidity/disease mappings are compact teaching versions, not validated Charlson/Elixhauser implementations.
- Synthetic generation is **not** a formal privacy guarantee (no differential privacy).

## Acknowledgements & citation

Built on the structure of [MIMIC-IV](https://physionet.org/content/mimiciv/) (Johnson et al., PhysioNet). If you use real MIMIC-IV, complete PhysioNet credentialing, sign the data use agreement, and cite the dataset. Cite this repository via [CITATION.cff](CITATION.cff).

## License

Code and synthetic data: [MIT](LICENSE). This license does **not** cover MIMIC-IV itself, which is governed by the PhysioNet Credentialed Health Data License.
