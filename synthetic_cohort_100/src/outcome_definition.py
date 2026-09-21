from __future__ import annotations
import pandas as pd

def compute_readmission(admissions:pd.DataFrame,index_hadm:set[int])->pd.DataFrame:
    a=admissions.copy(); a["admittime"]=pd.to_datetime(a.admittime); a["dischtime"]=pd.to_datetime(a.dischtime); rows=[]
    for hadm in sorted(index_hadm):
        row=a[a.hadm_id==hadm].iloc[0]; future=a[(a.subject_id==row.subject_id)&(a.admittime>row.dischtime)].sort_values("admittime"); days=(future.admittime.iloc[0]-row.dischtime).total_seconds()/86400 if len(future) else None; rows.append({"hadm_id":hadm,"readmitted_30d":int(days is not None and 0<days<=30)})
    return pd.DataFrame(rows)

