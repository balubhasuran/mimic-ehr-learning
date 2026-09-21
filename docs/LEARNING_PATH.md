# Learning path (about one day)

| Block | Time | Notebook / section | Goal |
|---|---|---|---|
| 1 | 30 min | 01 §1 | Explain what a row means in `patients`, `admissions`, `labevents`; identify the keys. |
| 2 | 45 min | 01 §2–4 | Build one-row-per-admission data with safe joins; run quality checks. |
| 3 | 45 min | 01 §3, `config/leakage_exclusions.yaml` | Decide which variables are available at prediction time. |
| 4 | 45 min | 01 §5–6 | EDA, missingness, preprocessing pipelines. |
| 5 | 60 min | 01 §7–9 | LOS and readmission models; why the dummy baseline sometimes wins; calibration. |
| 6 | 45 min | 01 §10–12 | Permutation importance, SHAP, coefficients. |
| 7 | 60 min | 02, 03 | Larger (1,000) dataset: mortality with rare events; LOS with skew. |
| 8 | 30 min | 00 (read-only) | See how a real-data source table is constructed. |

## Exercises

**Beginner**
1. Count admissions by age group (18–39, 40–59, 60–79, 80+).
2. Find the three most prevalent comorbidities.
3. Change a left join to an inner join. How many admissions disappear, and who are they?

**Intermediate**
4. Add a predictor from `labevents` (first 24 h) and check it cannot leak the outcome.
5. Replace random splits with grouped splits by patient. Does performance change?
6. Compare accuracy vs. AUPRC on the mortality task; explain why accuracy misleads.

**Advanced**
7. Recalibrate the readmission model (Platt / isotonic) and compare Brier scores.
8. Evaluate subgroup performance by race group and insurance; write a short bias-review paragraph.
9. Propose a temporal validation design for a real EHR.

## Common mistakes checklist
- Treating each `labevents` row as an independent patient.
- Joining two one-to-many tables before aggregating (row explosion).
- Random row splits when one patient has several admissions.
- Using discharge-time information to predict at-admission outcomes.
- Reporting accuracy on a 2%-prevalence outcome.
- Concluding from a small n that a complex model is "better".
