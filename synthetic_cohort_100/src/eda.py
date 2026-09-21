from __future__ import annotations
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMORB=["heart_failure","hypertension","metastatic_cancer","obesity","alcohol_abuse","depression","paralysis","diabetes","weight_loss","drug_abuse","chronic_kidney_disease","chronic_pulmonary_disease","liver_disease"]

def run_eda(model:pd.DataFrame,tables:dict[str,pd.DataFrame],domains:dict,out:Path)->None:
    summary=pd.DataFrame([{"unique_patients":model.subject_id.nunique(),"admissions":len(tables["admissions"]),"icu_stays":len(tables["icustays"]),"predictor_variables":sum(map(len,domains.values())),"overall_missingness_pct":model.isna().mean().mean()*100,"median_hospital_los_days":model.hospital_los_days.median(),"los_iqr":model.hospital_los_days.quantile(.75)-model.hospital_los_days.quantile(.25),"readmission_count":model.readmitted_30d.sum(),"readmission_pct":model.readmitted_30d.mean()*100}]); summary.to_csv(out/"tables"/"cohort_summary.csv",index=False)
    demo=[]
    for col in ["gender","race"]:
        for level,count in model[col].value_counts().items(): demo.append({"variable":col,"level":level,"count":count,"percentage":count/len(model)*100})
    age=model.age; demo += [{"variable":"age_summary","level":k,"count":v,"percentage":np.nan} for k,v in {"mean":age.mean(),"std":age.std(),"median":age.median(),"q1":age.quantile(.25),"q3":age.quantile(.75)}.items()]
    groups=pd.cut(age,[17,39,59,79,120],labels=["18-39","40-59","60-79","80+"]); demo += [{"variable":"age_group","level":k,"count":v,"percentage":v/len(model)*100} for k,v in groups.value_counts().sort_index().items()]; pd.DataFrame(demo).to_csv(out/"tables"/"demographic_summary.csv",index=False)
    prev=pd.DataFrame({"comorbidity":COMORB,"count":[model[c].sum() for c in COMORB],"prevalence_pct":[model[c].mean()*100 for c in COMORB]}).sort_values("prevalence_pct",ascending=False); prev.to_csv(out/"tables"/"comorbidity_prevalence.csv",index=False)
    clinical=[c for c in model.select_dtypes(include="number") if c not in ["subject_id","hadm_id","stay_id","readmitted_30d"]+COMORB]; model[clinical].describe().T.to_csv(out/"tables"/"clinical_variable_summary.csv"); pd.DataFrame({"variable":model.columns,"missing_count":model.isna().sum(),"missing_pct":model.isna().mean()*100}).to_csv(out/"tables"/"missingness_summary.csv",index=False); pd.DataFrame([{"domain":k,"variable_count":len(v)} for k,v in domains.items()]).to_csv(out/"tables"/"domain_summary.csv",index=False)
    combos=model[COMORB].apply(lambda r:" + ".join(sorted([c for c in COMORB if r[c]])) or "None",axis=1).value_counts().head(15).rename_axis("combination").reset_index(name="count"); combos.to_csv(out/"tables"/"common_comorbidity_combinations.csv",index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4)); model.age.hist(ax=axes[0],bins=15); axes[0].set_title("Age distribution"); model.gender.value_counts().plot.bar(ax=axes[1]); axes[1].set_title("Gender"); model.race.value_counts().plot.bar(ax=axes[2]); axes[2].set_title("Race"); fig.tight_layout(); fig.savefig(out/"figures"/"demographics.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5)); prev.sort_values("prevalence_pct").plot.barh(x="comorbidity",y="prevalence_pct",legend=False,ax=axes[0]); axes[0].set_title("Comorbidity prevalence (%)"); model.comorbidity_count.value_counts().sort_index().plot.bar(ax=axes[1]); axes[1].set_title("Comorbidity count"); fig.tight_layout(); fig.savefig(out/"figures"/"comorbidities.png",dpi=180); plt.close(fig)
    corr=model[COMORB].corr(); fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(corr,vmin=-1,vmax=1,cmap="coolwarm"); ax.set_xticks(range(len(COMORB)),COMORB,rotation=90,fontsize=7); ax.set_yticks(range(len(COMORB)),COMORB,fontsize=7); fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(out/"figures"/"comorbidity_cooccurrence_heatmap.png",dpi=180); plt.close(fig)
    miss=model.isna().mean().sort_values(ascending=False).head(30); fig,ax=plt.subplots(figsize=(9,6)); miss.sort_values().plot.barh(ax=ax); ax.set_title("Top missingness rates"); fig.tight_layout(); fig.savefig(out/"figures"/"missingness.png",dpi=180); plt.close(fig)
    report=["# Exploratory data analysis report","",f"- 100 unique synthetic patients and {len(tables['admissions'])} total admissions.",f"- {len(tables['icustays'])} ICU stays.",f"- Median hospital LOS: {model.hospital_los_days.median():.2f} days (IQR {model.hospital_los_days.quantile(.25):.2f}–{model.hospital_los_days.quantile(.75):.2f}).",f"- Readmission: {model.readmitted_30d.sum()} ({model.readmitted_30d.mean()*100:.1f}%).",f"- Mean comorbidity count: {model.comorbidity_count.mean():.2f}; median {model.comorbidity_count.median():.1f}; range {model.comorbidity_count.min()}–{model.comorbidity_count.max()}.","","All findings describe a small synthetic demonstration cohort and are not clinically validated."]; (out/"reports"/"eda_report.md").write_text("\n".join(report),encoding="utf-8")
