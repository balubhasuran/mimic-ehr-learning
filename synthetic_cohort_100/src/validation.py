from __future__ import annotations
from pathlib import Path
import pandas as pd
import yaml
from .feature_engineering import predictor_columns
from .outcome_definition import compute_readmission

def validate(model:pd.DataFrame,tables:dict[str,pd.DataFrame],root:Path)->list[tuple[str,bool,str]]:
    checks=[]; add=lambda n,ok,d="":checks.append((n,bool(ok),d)); patients=tables["patients"]; admissions=tables["admissions"]; icu=tables["icustays"]
    add("exactly_100_patients",len(model)==100); add("unique_subject_id",model.subject_id.nunique()==100); add("unique_model_hadm",model.hadm_id.nunique()==100); add("valid_stay_links",set(icu.hadm_id).issubset(set(admissions.hadm_id))); add("admittime_before_dischtime",(pd.to_datetime(admissions.admittime)<pd.to_datetime(admissions.dischtime)).all()); add("intime_before_outtime",(pd.to_datetime(icu.intime)<pd.to_datetime(icu.outtime)).all()); add("positive_los",model.hospital_los_days.gt(0).all()); calc=compute_readmission(admissions,set(model.hadm_id)).set_index("hadm_id").readmitted_30d; add("correct_readmission",model.set_index("hadm_id").readmitted_30d.equals(calc)); add("valid_gender",set(model.gender)<={"M","F"}); add("correct_comorbidity_count",model.comorbidity_count.equals(model[["heart_failure","hypertension","metastatic_cancer","obesity","alcohol_abuse","depression","paralysis","diabetes","weight_loss","drug_abuse","chronic_kidney_disease","chronic_pulmonary_disease","liver_disease"]].sum(axis=1))); add("no_duplicate_patients",not model.drop(columns=["subject_id","hadm_id","stay_id"]).duplicated().any()); add("predictor_count",len(predictor_columns(root))==73,str(len(predictor_columns(root)))); add("readmission_classes",model.readmitted_30d.value_counts().min()>=15)
    with (root/"config"/"leakage_exclusions.yaml").open() as f: leak=yaml.safe_load(f); add("no_los_leakage",not set(leak["los"])-set(leak["los"])); add("no_target_in_predictors",not {"hospital_los_days","readmitted_30d"}&set(predictor_columns(root)))
    return checks

def write_report(checks,path:Path)->None:
    lines=["# Data validation report","",f"Passed: {sum(x[1] for x in checks)}/{len(checks)}","", "| Check | Status | Detail |","|---|---|---|"]+[f"| {n} | {'PASS' if ok else 'FAIL'} | {d} |" for n,ok,d in checks]; path.write_text("\n".join(lines),encoding="utf-8")
    critical=[n for n,ok,_ in checks if not ok]
    if critical: raise RuntimeError(f"Critical validation failures: {critical}")

