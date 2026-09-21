from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,brier_score_loss,confusion_matrix,RocCurveDisplay,PrecisionRecallDisplay,ConfusionMatrixDisplay
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

def prep(x):
    num=list(x.select_dtypes(include="number")); cat=[c for c in x if c not in num]
    return ColumnTransformer([("n",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("c",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)])

def run_readmission(model:pd.DataFrame,features:list[str],out:Path,seed=42):
    x=model[features]; y=model.readmitted_30d.to_numpy(); cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=seed)
    models={"Dummy prior":DummyClassifier(strategy="prior"),"Logistic regression":LogisticRegression(max_iter=2000,class_weight="balanced",random_state=seed),"Random Forest":RandomForestClassifier(n_estimators=120,min_samples_leaf=3,class_weight="balanced",random_state=seed,n_jobs=1),"HistGradientBoosting":HistGradientBoostingClassifier(max_iter=120,random_state=seed)}
    rows=[]; allprob={}
    for name,est in models.items():
        sums=np.zeros(len(y)); counts=np.zeros(len(y)); fold=[]
        for tr,te in cv.split(x,y):
            pipe=Pipeline([("preprocess",prep(x.iloc[tr])),("model",est)]); pipe.fit(x.iloc[tr],y[tr]); pr=pipe.predict_proba(x.iloc[te])[:,1]; p=(pr>=.5).astype(int); sums[te]+=pr; counts[te]+=1; tn,fp,fn,tp=confusion_matrix(y[te],p,labels=[0,1]).ravel(); fold.append([roc_auc_score(y[te],pr),average_precision_score(y[te],pr),accuracy_score(y[te],p),balanced_accuracy_score(y[te],p),precision_score(y[te],p,zero_division=0),recall_score(y[te],p,zero_division=0),tn/max(tn+fp,1),f1_score(y[te],p,zero_division=0),brier_score_loss(y[te],pr)])
        vals=np.array(fold); rows.append(dict(zip(["model","auroc_mean","auprc_mean","accuracy_mean","balanced_accuracy_mean","precision_mean","sensitivity_mean","specificity_mean","f1_mean","brier_mean"],[name,*vals.mean(axis=0)]))); allprob[name]=sums/counts
    metrics=pd.DataFrame(rows).sort_values(["auroc_mean","auprc_mean"],ascending=False); best=metrics.iloc[0].model; prob=allprob[best]; pred=(prob>=.5).astype(int); pd.DataFrame({"subject_id":model.subject_id,"observed_readmitted_30d":y,"predicted_probability":prob,"predicted_class":pred,"model":best}).to_csv(out/"predictions"/"readmission_cross_validated_predictions.csv",index=False)
    pipe=Pipeline([("preprocess",prep(x)),("model",models[best])]); pipe.fit(x,y); imp=permutation_importance(pipe,x,y,n_repeats=5,random_state=seed,scoring="roc_auc"); importance=pd.DataFrame({"feature":features,"importance":imp.importances_mean}).sort_values("importance",ascending=False); importance.to_csv(out/"tables"/"readmission_feature_importance.csv",index=False)
    logistic=Pipeline([("preprocess",prep(x)),("model",models["Logistic regression"])]); logistic.fit(x,y)
    names=logistic.named_steps["preprocess"].get_feature_names_out(); coef=logistic.named_steps["model"].coef_[0]
    pd.DataFrame({"transformed_feature":names,"log_odds_coefficient":coef,"direction":np.where(coef>=0,"higher readmission odds","lower readmission odds")}).sort_values("log_odds_coefficient",key=abs,ascending=False).to_csv(out/"tables"/"readmission_logistic_coefficients.csv",index=False)
    fig,axes=plt.subplots(1,3,figsize=(15,4)); RocCurveDisplay.from_predictions(y,prob,ax=axes[0]); PrecisionRecallDisplay.from_predictions(y,prob,ax=axes[1]); ConfusionMatrixDisplay.from_predictions(y,pred,ax=axes[2]); fig.tight_layout(); fig.savefig(out/"figures"/"readmission_performance.png",dpi=180); plt.close(fig)
    frac,mean=calibration_curve(y,prob,n_bins=5); fig,ax=plt.subplots(); ax.plot(mean,frac,"o-"); ax.plot([0,1],[0,1],"--",color="black"); ax.set(xlabel="Mean predicted probability",ylabel="Observed fraction",title="Readmission calibration"); fig.tight_layout(); fig.savefig(out/"figures"/"readmission_calibration.png",dpi=180); plt.close(fig)
    top=importance.head(10).sort_values("importance"); fig,ax=plt.subplots(figsize=(8,5)); ax.barh(top.feature,top.importance); ax.set_title("Readmission permutation importance"); fig.tight_layout(); fig.savefig(out/"figures"/"readmission_permutation_importance.png",dpi=180); plt.close(fig)
    metrics.to_csv(out/"tables"/"readmission_model_metrics.csv",index=False); return metrics,best,importance
