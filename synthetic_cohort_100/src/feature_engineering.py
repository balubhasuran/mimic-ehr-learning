from __future__ import annotations
from pathlib import Path
import pandas as pd
import yaml
from .synthetic_generation import LABS,VITALS,COMORBIDITY_CODES

def load_config(root:Path)->dict:
    with (root/"config"/"feature_config.yaml").open(encoding="utf-8") as f: return yaml.safe_load(f)

def predictor_columns(root:Path)->list[str]:
    cfg=load_config(root); return [c for values in cfg["predictors"].values() for c in values]

def modeling_table(latent:pd.DataFrame,root:Path)->pd.DataFrame:
    predictors=predictor_columns(root); missing=[c for c in predictors if c not in latent]
    if missing: raise ValueError(f"Configured features not generated: {missing}")
    n=len(latent); out=latent[predictors+["hospital_los_days","readmitted_30d"]].copy()
    stays=pd.Series(range(70000001,70000001+n),dtype="Int64")
    stays[latent["icu_admission_indicator"].eq(0).to_numpy()]=pd.NA
    out.insert(0,"stay_id",stays); out.insert(0,"hadm_id",range(80000001,80000001+n)); out.insert(0,"subject_id",range(90000001,90000001+n)); return out

def feature_dictionary(root:Path)->pd.DataFrame:
    cfg=load_config(root); rows=[]
    for domain,features in cfg["predictors"].items():
        for f in features:
            table=column=item=label=unit=aggregation=""; window="first 24 hours"; missing="median imputation in model pipeline"
            if domain=="demographics":
                mapping={"age":("patients/admissions","anchor_age, anchor_year, admittime"),"gender":("patients","gender"),"race":("admissions","race"),"insurance":("admissions","insurance"),"marital_status":("admissions","marital_status"),"admission_type":("admissions","admission_type"),"admission_location":("admissions","admission_location"),"emergency_admission":("admissions","admission_type")}; table,column=mapping[f]; label=f.replace("_"," ").title(); aggregation="direct/derived at admission"; window="at admission"
            elif domain=="vitals":
                base=next(k for k in VITALS if f.startswith(k)); item,label,unit=VITALS[base]; table="chartevents"; column="valuenum, charttime, itemid"; aggregation="first" if f.endswith("first") else ("minimum" if f.endswith("min") else "maximum")
            elif domain=="comorbidities":
                table="diagnoses_icd"; column="icd_code, icd_version"; label=f.replace("_"," ").title(); aggregation="binary ICD-9/10 mapping" if f!="comorbidity_count" else "sum of 13 binary indicators"; item=str(COMORBIDITY_CODES.get(f,"documented mapping set")); window="diagnoses available by index discharge"; missing="0 when no mapped diagnosis"
            elif domain=="laboratory_chart":
                if f=="pf_ratio_first": table="labevents/chartevents"; column="valuenum, itemid"; label="PaO2/FiO2 ratio"; unit="ratio"; aggregation="first PaO2 divided by contemporaneous FiO2"
                else: item,label,unit=LABS[f]; table="labevents"; column="valuenum, charttime, itemid"; aggregation="first valid value"
            else:
                maps={"gcs_motor_first":("chartevents","valuenum, itemid",223901,"GCS - Motor Response",None),"gcs_verbal_first":("chartevents","valuenum, itemid",223900,"GCS - Verbal Response",None),"gcs_eyes_first":("chartevents","valuenum, itemid",220739,"GCS - Eye Opening",None),"gcs_total_first":("chartevents","valuenum, itemid","223901/223900/220739","Total GCS",None),"urine_output_24h":("chartevents","valuenum, itemid",226559,"Foley","mL"),"current_service":("services","curr_service","","Current service",None),"previous_service":("services","prev_service","","Previous service",None),"surgery_indicator":("services/procedures_icd","curr_service, icd_code","","Surgical service/procedure",None),"mechanical_ventilation_indicator":("chartevents","itemid","223848/223849/229314","Ventilator Type/Mode",None),"vasopressor_indicator":("inputevents","itemid","221906/221289/221749/222315","Norepinephrine/Epinephrine/Phenylephrine/Vasopressin",None),"icu_admission_indicator":("icustays","stay_id","","Any ICU stay",None),"procedure_count":("procedures_icd","hadm_id","","Number of procedures",None),"service_transfer_count":("services","transfertime","","Number of service transfers",None)}; table,column,item,label,unit=maps[f]; aggregation="first/sum/count/indicator as named"; window="first 24 hours for LOS; through discharge for readmission"
            rows.append({"feature_name":f,"domain":domain,"source_table":table,"source_column":column,"itemid_or_icd_code":item,"official_label":label,"unit":unit,"aggregation":aggregation,"observation_window":window,"missing_value_rule":missing,"used_for_los":True,"used_for_readmission":True,"exclusion_reason":""})
    return pd.DataFrame(rows)
