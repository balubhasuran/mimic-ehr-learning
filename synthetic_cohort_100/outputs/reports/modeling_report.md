# Modeling report

## Length of stay
Best model by repeated-CV MAE: **Dummy mean [log1p]**. MAE: 1.767 days; RMSE: 2.374; R-squared: -0.130. Raw and log1p target scales were evaluated, with predictions converted back to days.
Top predictors: age, chronic_kidney_disease, hematocrit_first, hemoglobin_first, wbc_first, bun_first, sodium_first, platelet_count_first, lactate_first, creatinine_first
Elastic Net coefficient directions are in `los_elastic_net_coefficients.csv`.

## Thirty-day readmission
Best model by repeated-CV AUROC/AUPRC: **HistGradientBoosting**. AUROC: 0.938; AUPRC: 0.832.
Top predictors: chronic_kidney_disease, comorbidity_count, hematocrit_first, age, lactate_first, hemoglobin_first, wbc_first, bun_first, sodium_first, platelet_count_first
Logistic-regression coefficient directions are in `readmission_logistic_coefficients.csv`.

Permutation importance and coefficient direction describe model behavior, not causality. Results are unstable because the cohort contains only 100 synthetic patients.