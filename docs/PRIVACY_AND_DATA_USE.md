# Privacy and data use

## What is in this repository
Only **synthetic** data and aggregate summaries. No MIMIC-IV rows, identifiers, or timestamps are included. Synthetic IDs (`SYN0001`, newly generated `subject_id` / `hadm_id` / `stay_id`) do not correspond to real patients.

Two files derive from the real cohort but contain only counts and summary statistics: `data/profiles_1000/cohort_flow.csv` and `data/profiles_1000/source_vs_synthetic_summary.csv`.

## How privacy was considered
- Distribution learning with perturbation of continuous values.
- Candidate synthetic profiles rejected when too close to a private real holdout (`generation_quality_report.csv`).
- Exact-duplicate and nearest-neighbour checks (`privacy_similarity_report.csv`): 0 exact and 0 near duplicates at the chosen threshold. The minimum normalized distance (about 0.025) sits right at that threshold, so treat this as a sanity screen, not a guarantee.

**This is not a formal privacy guarantee** (no differential privacy). Do not attempt re-identification.

## If you work with real MIMIC-IV
1. Complete CITI training and obtain PhysioNet credentialed access.
2. Sign the data use agreement; do not share the data or patient-level derivatives.
3. Keep real data outside this repository (`.gitignore` blocks common patterns, but you are responsible).
4. Do not post MIMIC-derived patient-level data, or notebook outputs containing real rows, to GitHub or to third-party LLM services.

## Clinical use
Nothing here is validated for clinical decision-making.
