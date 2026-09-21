from pathlib import Path
import yaml
from src.feature_engineering import predictor_columns
def test_no_targets_or_leakage_columns():
    features=set(predictor_columns(Path(".")))
    assert not {"hospital_los_days","readmitted_30d","dischtime","outtime","los","discharge_location"} & features
    assert len(features)==73

