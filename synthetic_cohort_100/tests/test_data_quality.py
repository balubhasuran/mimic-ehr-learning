from pathlib import Path
import pandas as pd
def test_modeling_quality():
    p=Path("data/synthetic_mimic_100_modeling.csv"); assert p.exists()
    d=pd.read_csv(p); assert len(d)==100 and d.subject_id.nunique()==100 and d.hadm_id.nunique()==100
    assert d.hospital_los_days.gt(0).all()
    flags=["heart_failure","hypertension","metastatic_cancer","obesity","alcohol_abuse","depression","paralysis","diabetes","weight_loss","drug_abuse","chronic_kidney_disease","chronic_pulmonary_disease","liver_disease"]
    assert d.comorbidity_count.equals(d[flags].sum(axis=1))

def test_no_exact_source_profile_match():
    privacy=pd.read_csv("outputs/tables/privacy_similarity_report.csv")
    assert privacy["exact_common_field_matches"].iloc[0]==0
