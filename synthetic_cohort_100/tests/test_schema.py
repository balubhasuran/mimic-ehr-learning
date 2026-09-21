import os
from pathlib import Path
import pytest
from src.schema_inspection import inspect_schemas

MIMIC_ROOT = os.environ.get("MIMIC_IV_ROOT")
pytestmark = pytest.mark.skipif(not MIMIC_ROOT, reason="Set MIMIC_IV_ROOT to a credentialed MIMIC-IV copy to run schema tests")

def test_required_schemas():
    _,_,schemas=inspect_schemas(Path(MIMIC_ROOT))
    assert {"subject_id","anchor_age"}.issubset(schemas["patients"])
    assert {"subject_id","hadm_id","admittime","dischtime"}.issubset(schemas["admissions"])
    assert {"itemid","charttime","valuenum","valueuom"}.issubset(schemas["labevents"])
