from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error,mean_squared_error,median_absolute_error,r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

def prep(x):
    num=list(x.select_dtypes(include="number")); cat=[c for c in x if c not in num]
    return ColumnTransformer([("n",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),("c",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False))]),cat)])

def run_los(model:pd.DataFrame,features:list[str],out:Path,seed=42):
    x=model[features]; y=model.hospital_los_days.to_numpy(); cv=RepeatedKFold(n_splits=5,n_repeats=3,random_state=seed)
    models={"Dummy mean":DummyRegressor(),"Elastic Net":ElasticNet(alpha=.05,l1_ratio=.3,max_iter=10000,random_state=seed),"Random Forest":RandomForestRegressor(n_estimators=120,min_samples_leaf=3,random_state=seed,n_jobs=1),"HistGradientBoosting":HistGradientBoostingRegressor(max_iter=120,random_state=seed)}
    rows=[]; allpred={}
    for scale in ("raw","log1p"):
        fit_y=y if scale=="raw" else np.log1p(y)
        for name,est in models.items():
            sums=np.zeros(len(y)); counts=np.zeros(len(y)); fold=[]
            for tr,te in cv.split(x):
                pipe=Pipeline([("preprocess",prep(x.iloc[tr])),("model",est)]); pipe.fit(x.iloc[tr],fit_y[tr])
                p=pipe.predict(x.iloc[te]); p=np.maximum(p if scale=="raw" else np.expm1(p),0)
                sums[te]+=p; counts[te]+=1; fold.append([mean_absolute_error(y[te],p),mean_squared_error(y[te],p)**.5,r2_score(y[te],p),median_absolute_error(y[te],p)])
            vals=np.array(fold); key=f"{name} [{scale}]"; rows.append({"model":name,"target_scale":scale,"mae_mean":vals[:,0].mean(),"mae_sd":vals[:,0].std(),"rmse_mean":vals[:,1].mean(),"r2_mean":vals[:,2].mean(),"median_ae_mean":vals[:,3].mean()}); allpred[key]=sums/counts
    metrics=pd.DataFrame(rows).sort_values("mae_mean"); best_name=str(metrics.iloc[0].model); best_scale=str(metrics.iloc[0].target_scale); best=f"{best_name} [{best_scale}]"; pred=allpred[best]
    pd.DataFrame({"subject_id":model.subject_id,"observed_hospital_los_days":y,"predicted_hospital_los_days":pred,"model":best_name,"target_scale":best_scale}).to_csv(out/"predictions"/"los_cross_validated_predictions.csv",index=False)
    fit_y=y if best_scale=="raw" else np.log1p(y)
    pipe=Pipeline([("preprocess",prep(x)),("model",models[best_name])]); pipe.fit(x,fit_y)
    def score_days(estimator, xx, yy): return -mean_absolute_error(yy,np.maximum(estimator.predict(xx) if best_scale=="raw" else np.expm1(estimator.predict(xx)),0))
    imp=permutation_importance(pipe,x,y,n_repeats=5,random_state=seed,scoring=score_days); importance=pd.DataFrame({"feature":features,"importance":imp.importances_mean}).sort_values("importance",ascending=False); importance.to_csv(out/"tables"/"los_feature_importance.csv",index=False)
    linear=Pipeline([("preprocess",prep(x)),("model",models["Elastic Net"])]); linear.fit(x,np.log1p(y))
    names=linear.named_steps["preprocess"].get_feature_names_out(); coef=linear.named_steps["model"].coef_
    pd.DataFrame({"transformed_feature":names,"coefficient_log1p_los":coef,"direction":np.where(coef>=0,"higher LOS","lower LOS")}).sort_values("coefficient_log1p_los",key=abs,ascending=False).to_csv(out/"tables"/"los_elastic_net_coefficients.csv",index=False)
    fig,ax=plt.subplots(); ax.scatter(y,pred,alpha=.7); lim=max(y.max(),pred.max()); ax.plot([0,lim],[0,lim],"--",color="black"); ax.set(xlabel="Observed LOS (days)",ylabel="Cross-validated predicted LOS (days)",title=f"LOS observed vs predicted — {best}"); fig.tight_layout(); fig.savefig(out/"figures"/"los_observed_vs_predicted.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(); ax.scatter(pred,y-pred,alpha=.7); ax.axhline(0,ls="--",color="black"); ax.set(xlabel="Predicted LOS",ylabel="Residual",title="LOS residuals"); fig.tight_layout(); fig.savefig(out/"figures"/"los_residuals.png",dpi=180); plt.close(fig)
    top=importance.head(10).sort_values("importance"); fig,ax=plt.subplots(figsize=(8,5)); ax.barh(top.feature,top.importance); ax.set_title("LOS permutation importance"); fig.tight_layout(); fig.savefig(out/"figures"/"los_permutation_importance.png",dpi=180); plt.close(fig)
    metrics.to_csv(out/"tables"/"los_model_metrics.csv",index=False); return metrics,best,importance
