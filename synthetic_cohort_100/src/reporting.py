from __future__ import annotations
from pathlib import Path
import pandas as pd

def write_reports(model:pd.DataFrame,los:pd.DataFrame,readmit:pd.DataFrame,best_los:str,best_readmit:str,los_imp:pd.DataFrame,read_imp:pd.DataFrame,out:Path)->None:
    text=[
        "# Modeling report","",
        f"## Length of stay\nBest model by repeated-CV MAE: **{best_los}**. MAE: {los.iloc[0].mae_mean:.3f} days; RMSE: {los.iloc[0].rmse_mean:.3f}; R-squared: {los.iloc[0].r2_mean:.3f}. Raw and log1p target scales were evaluated, with predictions converted back to days.",
        "Top predictors: "+", ".join(los_imp.head(10).feature),
        "Elastic Net coefficient directions are in `los_elastic_net_coefficients.csv`.","",
        f"## Thirty-day readmission\nBest model by repeated-CV AUROC/AUPRC: **{best_readmit}**. AUROC: {readmit.iloc[0].auroc_mean:.3f}; AUPRC: {readmit.iloc[0].auprc_mean:.3f}.",
        "Top predictors: "+", ".join(read_imp.head(10).feature),
        "Logistic-regression coefficient directions are in `readmission_logistic_coefficients.csv`.","",
        "Permutation importance and coefficient direction describe model behavior, not causality. Results are unstable because the cohort contains only 100 synthetic patients.",
    ]
    (out/"reports"/"modeling_report.md").write_text("\n".join(text),encoding="utf-8")
    final=["# Final summary","",f"- Synthetic patients: {model.subject_id.nunique()}",f"- Index admissions: {model.hadm_id.nunique()}",f"- Predictor variables: {len(los_imp)}",f"- Readmission prevalence: {model.readmitted_30d.mean()*100:.1f}%",f"- Median hospital LOS: {model.hospital_los_days.median():.2f} days",f"- Best LOS model: {best_los} (CV MAE {los.iloc[0].mae_mean:.3f} days)",f"- Best readmission model: {best_readmit} (CV AUROC {readmit.iloc[0].auroc_mean:.3f}; AUPRC {readmit.iloc[0].auprc_mean:.3f})","","Educational demonstration only. Synthetic data and models are not clinically validated and must not be used for clinical decisions."]
    (out/"reports"/"final_summary.md").write_text("\n".join(final),encoding="utf-8")
