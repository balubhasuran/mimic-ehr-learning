from __future__ import annotations
from pathlib import Path
import pandas as pd

EXPECTED={"patients","admissions","icustays","diagnoses_icd","procedures_icd","services","labevents","chartevents","inputevents"}

def resolve_mimic_root(root:Path)->Path:
    for p in [root,*[x.parent for x in root.rglob("hosp") if x.is_dir()]]:
        if (p/"hosp").is_dir() and (p/"icu").is_dir(): return p.resolve()
    raise FileNotFoundError(f"MIMIC-IV hosp/icu folders not found beneath {root}")

def locate(base:Path,table:str)->Path:
    module="icu" if table in {"icustays","chartevents","inputevents"} else "hosp"
    for name in [f"{table}.csv.gz",f"{table}.csv",f"{table}.parquet"]:
        p=base/module/name
        if p.is_file(): return p
    raise FileNotFoundError(f"Required table unavailable: {module}/{table}")

def header(path:Path)->list[str]:
    return list(pd.read_parquet(path).columns) if path.suffix==".parquet" else list(pd.read_csv(path,nrows=0).columns)

def inspect_schemas(root:Path)->tuple[Path,dict[str,Path],dict[str,list[str]]]:
    base=resolve_mimic_root(root); paths={t:locate(base,t) for t in EXPECTED}; return base,paths,{t:header(p) for t,p in paths.items()}
