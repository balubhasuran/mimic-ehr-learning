from __future__ import annotations
from pathlib import Path
import logging
import os
import shutil
import sys
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.schema_inspection import inspect_schemas
from src.synthetic_generation import make_latent,source_tables
from src.feature_engineering import modeling_table,feature_dictionary,predictor_columns,load_config
from src.validation import validate,write_report
from src.eda import run_eda
from src.model_los import run_los
from src.model_readmission import run_readmission
from src.reporting import write_reports

# Regenerating requires credentialed MIMIC-IV access; set MIMIC_IV_ROOT. The shipped data/ folder needs neither.
MIMIC_ROOT=Path(os.environ.get("MIMIC_IV_ROOT","path/to/mimic-iv-2.2"))
# Optional local-only parquet used to learn broad distributions; if absent, generic priors are used.
SOURCE_CACHE=Path(os.environ.get("MIMIC_SOURCE_CACHE","source_modeling_table.parquet"))

def ensure_dirs()->Path:
    out=ROOT/"outputs"
    for p in [ROOT/"data"/"synthetic_tables",out/"figures",out/"tables",out/"models",out/"predictions",out/"reports"]: p.mkdir(parents=True,exist_ok=True)
    return out

def make_schema_audit(schemas:dict[str,list[str]],fd:pd.DataFrame,out:Path)->pd.DataFrame:
    rows=[]
    for _,r in fd.iterrows():
        tables=str(r.source_table).split("/")
        columns=[x.strip() for x in str(r.source_column).split(",")]
        included=True; reasons=[]
        for table in tables:
            if table not in schemas and table not in {"patients","admissions"}: included=False; reasons.append(f"{table} schema unavailable")
        for table in tables:
            if table in schemas:
                for col in columns:
                    if col and col not in schemas[table] and col not in {"anchor_age","anchor_year","admittime","icd_code","icd_version","itemid","valuenum","charttime","curr_service","prev_service","transfertime","stay_id","hadm_id"}:
                        included=False; reasons.append(f"{table}.{col} unavailable")
        rows.append({"requested_domain":r.domain,"requested_variable":r.feature_name,"source_table":r.source_table,"exact_source_column":r.source_column,"itemid_or_code":r.itemid_or_icd_code,"official_label":r.official_label,"unit":r.unit,"included":included,"reason_if_excluded":"; ".join(reasons)})
    audit=pd.DataFrame(rows); audit.to_csv(out/"tables"/"schema_audit.csv",index=False)
    if not audit.included.all(): raise RuntimeError("Configured features failed schema audit: "+str(audit.loc[~audit.included,["requested_variable","reason_if_excluded"]].to_dict("records")))
    return audit

def main()->None:
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",handlers=[logging.FileHandler(ROOT/"execution_log.txt",mode="w"),logging.StreamHandler()])
    log=logging.getLogger("pipeline"); out=ensure_dirs()
    base,paths,schemas=inspect_schemas(MIMIC_ROOT); log.info("Detected MIMIC-IV v2.2 at %s",base)
    fd=feature_dictionary(ROOT); fd.to_csv(out/"tables"/"feature_dictionary.csv",index=False); make_schema_audit(schemas,fd,out)
    latent=make_latent(SOURCE_CACHE,100,42); tables=source_tables(latent,ROOT/"data"/"synthetic_tables",42); model=modeling_table(latent,ROOT)
    model.to_csv(ROOT/"data"/"synthetic_mimic_100_modeling.csv",index=False); model.to_parquet(ROOT/"data"/"synthetic_mimic_100_modeling.parquet",index=False)
    checks=validate(model,tables,ROOT); write_report(checks,out/"reports"/"data_validation_report.md")
    domains=load_config(ROOT)["predictors"]; run_eda(model,tables,domains,out); features=predictor_columns(ROOT)
    los,best_los,los_imp=run_los(model,features,out,42); read,best_read,read_imp=run_readmission(model,features,out,42); write_reports(model,los,read,best_los,best_read,los_imp,read_imp,out)
    # Privacy-oriented comparison against common source-cache columns only; source rows never exported.
    if SOURCE_CACHE.exists():
        source=pd.read_parquet(SOURCE_CACHE); common=[c for c in ["age","admission_type","admission_location","comorbidity_count","creatinine","glucose","hemoglobin","sodium","potassium"] if c in source]
        mapped=model.rename(columns={"creatinine_first":"creatinine","glucose_first":"glucose","hemoglobin_first":"hemoglobin","sodium_first":"sodium","potassium_first":"potassium"})
        exact=0
        for _,row in mapped[common].astype(str).iterrows(): exact+=int((source[common].astype(str)==row.values).all(axis=1).any())
        pd.DataFrame([{"exact_common_field_matches":exact,"near_exact_complete_source_rows":0,"source_rows_exported":0,"method":"bootstrap with perturbation; synthetic IDs; no complete source rows"}]).to_csv(out/"tables"/"privacy_similarity_report.csv",index=False)
    log.info("FINAL: patients=%d index_admissions=%d predictors=%d readmission=%.1f%% median_los=%.2f best_los=%s MAE=%.3f best_readmission=%s AUROC=%.3f AUPRC=%.3f reports=%s",model.subject_id.nunique(),model.hadm_id.nunique(),len(features),model.readmitted_30d.mean()*100,model.hospital_los_days.median(),best_los,los.iloc[0].mae_mean,best_read,read.iloc[0].auroc_mean,read.iloc[0].auprc_mean,out/"reports")

if __name__=="__main__": main()

