import pandas as pd
from src.outcome_definition import compute_readmission
def test_readmission_matches_source_tables():
    model=pd.read_csv("data/synthetic_mimic_100_modeling.csv")
    admissions=pd.read_csv("data/synthetic_tables/admissions.csv")
    computed=compute_readmission(admissions,set(model.hadm_id)).set_index("hadm_id").readmitted_30d
    assert model.set_index("hadm_id").readmitted_30d.equals(computed)
    assert 0.15 <= model.readmitted_30d.mean() <= 0.25

