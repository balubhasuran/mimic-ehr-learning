from __future__ import annotations
from pathlib import Path
import logging
import numpy as np
import pandas as pd

LOG=logging.getLogger(__name__)

LABS={
"pao2_first":(50821,"pO2","mmHg"),"pco2_first":(50818,"pCO2","mmHg"),"ph_first":(50820,"pH","units"),
"bicarbonate_first":(50882,"Bicarbonate","mEq/L"),"creatinine_first":(50912,"Creatinine","mg/dL"),"lactate_first":(50813,"Lactate","mmol/L"),
"platelet_count_first":(51265,"Platelet Count","K/uL"),"sodium_first":(50983,"Sodium","mEq/L"),"bun_first":(51006,"Urea Nitrogen","mg/dL"),
"wbc_first":(51301,"White Blood Cells","K/uL"),"hemoglobin_first":(51222,"Hemoglobin","g/dL"),"hematocrit_first":(51221,"Hematocrit","%"),
"potassium_first":(50971,"Potassium","mEq/L"),"chloride_first":(50902,"Chloride","mEq/L"),"glucose_first":(50931,"Glucose","mg/dL"),
"albumin_first":(50862,"Albumin","g/dL"),"bilirubin_first":(50885,"Bilirubin, Total","mg/dL"),"inr_first":(51237,"INR(PT)","ratio")}

VITALS={
"heart_rate":(220045,"Heart Rate","bpm"),"systolic_bp":(220179,"Non Invasive Blood Pressure systolic","mmHg"),
"diastolic_bp":(220180,"Non Invasive Blood Pressure diastolic","mmHg"),"mean_arterial_pressure":(220181,"Non Invasive Blood Pressure mean","mmHg"),
"respiratory_rate":(220210,"Respiratory Rate","insp/min"),"spo2":(220277,"O2 saturation pulseoxymetry","%"),"temperature":(223762,"Temperature Celsius","°C"),
"gcs_motor":(223901,"GCS - Motor Response",None),"gcs_verbal":(223900,"GCS - Verbal Response",None),"gcs_eyes":(220739,"GCS - Eye Opening",None),"urine_output":(226559,"Foley","mL")}

COMORBIDITY_CODES={
"heart_failure":("I50",10),"hypertension":("I10",10),"metastatic_cancer":("C79",10),"obesity":("E66",10),"alcohol_abuse":("F10",10),
"depression":("F32",10),"paralysis":("G82",10),"diabetes":("E11",10),"weight_loss":("R63",10),"drug_abuse":("F11",10),
"chronic_kidney_disease":("N18",10),"chronic_pulmonary_disease":("J44",10),"liver_disease":("K74",10)}

def _source_seed(cache:Path,n:int,rng:np.random.Generator)->pd.DataFrame:
    if cache.exists(): return pd.read_parquet(cache).sample(n=n,replace=False,random_state=42).reset_index(drop=True)
    return pd.DataFrame(index=range(n))

def make_latent(cache:Path,n:int=100,seed:int=42)->pd.DataFrame:
    rng=np.random.default_rng(seed); s=_source_seed(cache,n,rng); x=pd.DataFrame(index=range(n))
    def take(c,default): return s[c].to_numpy() if c in s else default
    x["age"]=np.clip(take("age",rng.normal(64,17,n))+rng.normal(0,1.8,n),18,95).round()
    x["gender"]=take("sex",rng.choice(["Male","Female"],n)).astype(str).tolist(); x["gender"]=x.gender.map({"Male":"M","Female":"F"}).fillna(x.gender)
    x["race"]=take("race_group",rng.choice(["White","Black","Asian","Hispanic","Other_or_unknown"],n,p=[.55,.2,.08,.1,.07])).astype(str)
    x["insurance"]=pd.Series(take("insurance_group",rng.choice(["Medicare","Medicaid","Private","Other"],n))).astype(str).str.replace("_"," ")
    x["marital_status"]=pd.Series(take("marital_status",rng.choice(["Married","Single","Divorced","Widowed","Unknown"],n))).astype(str).str.replace("_"," ")
    x["admission_type"]=pd.Series(take("admission_type",rng.choice(["Emergency","Urgent","Elective"],n,p=[.65,.15,.2]))).astype(str).str.replace("_"," ")
    x["admission_location"]=pd.Series(take("admission_location",rng.choice(["Emergency Room","Physician Referral","Transfer From Hospital"],n))).astype(str).str.replace("_"," ")
    x["emergency_admission"]=x.admission_type.str.contains("Emergency|Urgent",case=False).astype(int)
    burden=np.clip(rng.poisson(2.2+0.025*np.maximum(x.age-50,0)),0,10)
    probs={"heart_failure":.10,"hypertension":.45,"metastatic_cancer":.05,"obesity":.25,"alcohol_abuse":.08,"depression":.18,"paralysis":.04,"diabetes":.25,"weight_loss":.10,"drug_abuse":.07,"chronic_kidney_disease":.18,"chronic_pulmonary_disease":.18,"liver_disease":.10}
    for c,p in probs.items(): x[c]=rng.binomial(1,np.clip(p+.025*burden,0,.75))
    x["comorbidity_count"]=x[list(probs)].sum(axis=1); severity=x.comorbidity_count+.8*x.emergency_admission+rng.normal(0,1,n)
    bases={"heart_rate":85,"systolic_bp":125,"diastolic_bp":72,"respiratory_rate":18,"spo2":96,"temperature":37}
    for v,b in bases.items():
        shift={"heart_rate":4,"systolic_bp":-2,"diastolic_bp":-1,"respiratory_rate":1.2,"spo2":-.8,"temperature":.12}[v]*severity
        first=b+shift+rng.normal(0,{"heart_rate":13,"systolic_bp":18,"diastolic_bp":12,"respiratory_rate":4,"spo2":2.5,"temperature":.7}[v],n)
        if v=="spo2": first=np.clip(first,65,100)
        if v=="temperature": first=np.clip(first,32,42)
        x[f"{v}_first"]=first; spread=np.abs(rng.normal({"heart_rate":14,"systolic_bp":18,"diastolic_bp":12,"respiratory_rate":5,"spo2":3,"temperature":.8}[v],2,n)); x[f"{v}_min"]=first-spread/2; x[f"{v}_max"]=first+spread/2
    x["mean_arterial_pressure_first"]=(x.systolic_bp_first+2*x.diastolic_bp_first)/3
    defaults={"pao2_first":90,"pco2_first":40,"ph_first":7.39,"bicarbonate_first":24,"creatinine_first":1.0,"lactate_first":1.7,"platelet_count_first":220,"sodium_first":139,"bun_first":18,"wbc_first":9,"hemoglobin_first":12.5,"hematocrit_first":38,"potassium_first":4.1,"chloride_first":103,"glucose_first":120,"albumin_first":3.7,"bilirubin_first":.8,"inr_first":1.1}
    srcmap={"bicarbonate_first":"bicarbonate","creatinine_first":"creatinine","lactate_first":"lactate","platelet_count_first":"platelet_count","sodium_first":"sodium","bun_first":"blood_urea_nitrogen","wbc_first":"white_blood_cell_count","hemoglobin_first":"hemoglobin","potassium_first":"potassium","chloride_first":"chloride","glucose_first":"glucose","albumin_first":"albumin","bilirubin_first":"bilirubin","inr_first":"inr"}
    sd={"pao2_first":25,"pco2_first":10,"ph_first":.08,"bicarbonate_first":5,"creatinine_first":.8,"lactate_first":1.1,"platelet_count_first":90,"sodium_first":5,"bun_first":14,"wbc_first":5,"hemoglobin_first":2,"hematocrit_first":6,"potassium_first":.6,"chloride_first":6,"glucose_first":55,"albumin_first":.7,"bilirubin_first":1.2,"inr_first":.4}
    for f,b in defaults.items(): x[f]=take(srcmap.get(f,""),b+rng.normal(0,sd[f],n)).astype(float)+rng.normal(0,.04*sd[f],n)
    x["hematocrit_first"]=np.where(np.isfinite(x.hemoglobin_first),x.hemoglobin_first*3+rng.normal(0,2,n),x.hematocrit_first)
    x["pf_ratio_first"]=np.clip(x.pao2_first/np.clip(rng.normal(.35+.015*severity,.12,n),.21,1),20,700)
    x["gcs_motor_first"]=np.clip(np.round(6-.18*severity+rng.normal(0,.5,n)),1,6); x["gcs_verbal_first"]=np.clip(np.round(5-.15*severity+rng.normal(0,.6,n)),1,5); x["gcs_eyes_first"]=np.clip(np.round(4-.12*severity+rng.normal(0,.5,n)),1,4); x["gcs_total_first"]=x.gcs_motor_first+x.gcs_verbal_first+x.gcs_eyes_first
    x["urine_output_24h"]=np.clip(rng.lognormal(np.log(1700)-.12*severity,.45,n),0,8000); x["current_service"]=rng.choice(["MED","CMED","SURG","CSURG","TRAUM","NMED"],n); x["previous_service"]=rng.choice(["","MED","SURG","ED"],n); x["surgery_indicator"]=x.current_service.str.contains("SURG").astype(int); x["mechanical_ventilation_indicator"]=rng.binomial(1,np.clip(.08+.04*severity,0,.65)); x["vasopressor_indicator"]=rng.binomial(1,np.clip(.05+.035*severity,0,.55)); x["icu_admission_indicator"]=rng.binomial(1,np.clip(.25+.05*severity,0,.85)); x["procedure_count"]=rng.poisson(np.clip(1+.35*severity,.2,8)); x["service_transfer_count"]=rng.poisson(np.clip(.5+.12*severity,.1,3))
    x["hospital_los_days"]=np.clip(rng.lognormal(np.log(2.2+.55*severity),.55,n),.3,45); readmit_p=1/(1+np.exp(-(-2.1+.16*x.comorbidity_count+.025*x.hospital_los_days+.35*x.chronic_kidney_disease))); x["readmitted_30d"]=rng.binomial(1,np.clip(readmit_p,.08,.45))
    # enforce workshop-friendly prevalence 15-25%
    target=20; order=np.argsort(readmit_p.to_numpy())[::-1]; x["readmitted_30d"]=0; x.loc[order[:target],"readmitted_30d"]=1
    # realistic missingness
    for f in list(LABS)+["pao2_first","pco2_first","pf_ratio_first","ph_first"]:
        if f in x: x.loc[rng.random(n)<(.12 if f not in {"pao2_first","pco2_first","pf_ratio_first"} else .45),f]=np.nan
    return x

def source_tables(x:pd.DataFrame,out:Path,seed:int=42)->dict[str,pd.DataFrame]:
    rng=np.random.default_rng(seed); out.mkdir(parents=True,exist_ok=True); n=len(x); subject=np.arange(90000001,90000001+n); hadm=np.arange(80000001,80000001+n); stay=np.arange(70000001,70000001+n); anchor_year=2150; admit=pd.Timestamp("2150-01-01")+pd.to_timedelta(rng.integers(0,300,n),unit="D")
    patients=pd.DataFrame({"subject_id":subject,"gender":x.gender,"anchor_age":x.age.astype(int),"anchor_year":anchor_year,"anchor_year_group":"2147 - 2152","dod":pd.NaT})
    admissions=[]
    for i in range(n):
        discharge=admit[i]+pd.Timedelta(days=float(x.hospital_los_days.iloc[i])); admissions.append([subject[i],hadm[i],admit[i],discharge,pd.NaT,x.admission_type.iloc[i],"P00000",x.admission_location.iloc[i],"HOME",x.insurance.iloc[i],"ENGLISH",x.marital_status.iloc[i],x.race.iloc[i],admit[i]-pd.Timedelta(hours=2),admit[i]-pd.Timedelta(minutes=20),0])
        if x.readmitted_30d.iloc[i]:
            gap=int(rng.integers(2,29)); ra=discharge+pd.Timedelta(days=gap); admissions.append([subject[i],hadm[i]+1000,ra,ra+pd.Timedelta(days=float(np.clip(rng.lognormal(1,.5),.3,20))),pd.NaT,"URGENT","P00000","EMERGENCY ROOM","HOME",x.insurance.iloc[i],"ENGLISH",x.marital_status.iloc[i],x.race.iloc[i],ra-pd.Timedelta(hours=1),ra-pd.Timedelta(minutes=10),0])
    admissions=pd.DataFrame(admissions,columns=["subject_id","hadm_id","admittime","dischtime","deathtime","admission_type","admit_provider_id","admission_location","discharge_location","insurance","language","marital_status","race","edregtime","edouttime","hospital_expire_flag"])
    icu=[]
    for i in range(n):
        if x.icu_admission_indicator.iloc[i]:
            intime=admit[i]+pd.Timedelta(hours=float(rng.uniform(0,10))); ilos=min(float(x.hospital_los_days.iloc[i]),float(rng.uniform(.5,5))); icu.append([subject[i],hadm[i],stay[i],"Medical Intensive Care Unit (MICU)","Medical Intensive Care Unit (MICU)",intime,intime+pd.Timedelta(days=ilos),ilos])
    icu=pd.DataFrame(icu,columns=["subject_id","hadm_id","stay_id","first_careunit","last_careunit","intime","outtime","los"])
    dx=[]
    for i in range(n):
        seq=1
        for f,(code,ver) in COMORBIDITY_CODES.items():
            if x[f].iloc[i]: dx.append([subject[i],hadm[i],seq,code,ver]); seq+=1
        if seq==1: dx.append([subject[i],hadm[i],1,"R69",10])
    dx=pd.DataFrame(dx,columns=["subject_id","hadm_id","seq_num","icd_code","icd_version"])
    proc=[]
    for i in range(n):
        for j in range(int(x.procedure_count.iloc[i])): proc.append([subject[i],hadm[i],j+1,(admit[i]+pd.Timedelta(hours=12)).date(),"0W3P0ZZ",10])
    proc=pd.DataFrame(proc,columns=["subject_id","hadm_id","seq_num","chartdate","icd_code","icd_version"])
    services=[]
    for i in range(n):
        services.append([subject[i],hadm[i],admit[i],x.previous_service.iloc[i],x.current_service.iloc[i]])
        for j in range(int(x.service_transfer_count.iloc[i])): services.append([subject[i],hadm[i],admit[i]+pd.Timedelta(hours=8+j*4),x.current_service.iloc[i],x.current_service.iloc[i]])
    services=pd.DataFrame(services,columns=["subject_id","hadm_id","transfertime","prev_service","curr_service"])
    labs=[]; lid=1
    for i in range(n):
        for f,(item,label,unit) in LABS.items():
            if f in x and pd.notna(x[f].iloc[i]):
                chart=admit[i]+pd.Timedelta(hours=float(rng.uniform(0,20))); val=float(x[f].iloc[i]); labs.append([lid,subject[i],hadm[i],lid,item,"P00000",chart,chart+pd.Timedelta(minutes=20),f"{val:.3f}",val,unit,np.nan,np.nan,"normal","ROUTINE",""]); lid+=1
    labs=pd.DataFrame(labs,columns=["labevent_id","subject_id","hadm_id","specimen_id","itemid","order_provider_id","charttime","storetime","value","valuenum","valueuom","ref_range_lower","ref_range_upper","flag","priority","comments"])
    charts=[]
    for i in range(n):
        if not x.icu_admission_indicator.iloc[i]: continue
        for base,(item,label,unit) in VITALS.items():
            feature=f"{base}_first" if base!="urine_output" else "urine_output_24h"
            if feature not in x: continue
            val=float(x[feature].iloc[i]); chart=admit[i]+pd.Timedelta(hours=float(rng.uniform(0,18))); charts.append([subject[i],hadm[i],stay[i],999,chart,chart+pd.Timedelta(minutes=5),item,str(val),val,unit,0])
        if x.mechanical_ventilation_indicator.iloc[i]:
            chart=admit[i]+pd.Timedelta(hours=2)
            charts.append([subject[i],hadm[i],stay[i],999,chart,chart+pd.Timedelta(minutes=5),223849,"Assist Control",np.nan,"",0])
    charts=pd.DataFrame(charts,columns=["subject_id","hadm_id","stay_id","caregiver_id","charttime","storetime","itemid","value","valuenum","valueuom","warning"])
    inputs=[]; order=1
    for i in range(n):
        if x.vasopressor_indicator.iloc[i] and x.icu_admission_indicator.iloc[i]:
            start=admit[i]+pd.Timedelta(hours=2)
            inputs.append([subject[i],hadm[i],stay[i],999,start,start+pd.Timedelta(hours=6),start+pd.Timedelta(minutes=5),221906,4.0,"mg",0.08,"mcg/kg/min",order,order,"Continuous IV","","Main order parameter","Continuous infusion",75.0,4.0,"mg",0,0,"FinishedRunning",4.0,0.08]); order+=1
    inputs=pd.DataFrame(inputs,columns=["subject_id","hadm_id","stay_id","caregiver_id","starttime","endtime","storetime","itemid","amount","amountuom","rate","rateuom","orderid","linkorderid","ordercategoryname","secondaryordercategoryname","ordercomponenttypedescription","ordercategorydescription","patientweight","totalamount","totalamountuom","isopenbag","continueinnextdept","statusdescription","originalamount","originalrate"])
    tables={"patients":patients,"admissions":admissions,"icustays":icu,"diagnoses_icd":dx,"procedures_icd":proc,"services":services,"labevents":labs,"chartevents":charts,"inputevents":inputs}
    for name,df in tables.items(): df.to_csv(out/f"{name}.csv",index=False)
    return tables
