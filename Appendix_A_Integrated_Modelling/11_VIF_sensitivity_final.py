import json,joblib,numpy as np,pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,f1_score,matthews_corrcoef
from xgboost import XGBClassifier
from appendix_config import *
VIF_DIR.mkdir(exist_ok=True); df=pd.read_csv(DATA_FILE); sp=pd.read_csv(VAL_DIR/"01_fixed_split_ids.csv")
tr=df[df[ID_COL].isin(set(sp.loc[sp.split=="TRAIN",ID_COL]))].copy(); te=df[df[ID_COL].isin(set(sp.loc[sp.split=="TEST",ID_COL]))].copy()
imp=SimpleImputer(strategy="median"); X=pd.DataFrame(imp.fit_transform(tr[ALL_PREDICTORS]),columns=ALL_PREDICTORS); Xs=pd.DataFrame(StandardScaler().fit_transform(X),columns=X.columns)
def ivif(X,t):
 cols=list(X.columns); hist=[]
 while len(cols)>2:
  vals=pd.Series([variance_inflation_factor(X[cols].values,i) for i in range(len(cols))],index=cols); mx=float(vals.max()); var=str(vals.idxmax()); hist.append({"threshold":t,"n_predictors":len(cols),"max_vif":mx,"max_variable":var})
  if np.isfinite(mx) and mx<=t: break
  cols.remove(var)
 return cols,pd.DataFrame(hist)
dm=pd.read_csv(VAL_DIR/"10_environmental_domain_ablation_metrics.csv"); row=dm[dm.domain_set=="M+T+FV"].iloc[0]; pars=json.loads(row.best_params) if isinstance(row.best_params,str) else {}; clean={k.replace("model__",""):v for k,v in pars.items() if k.startswith("model__")}
rows=[]; hs=[]
for t in [10,5]:
 keep,h=ivif(Xs,t); hs.append(h); f=VIF_DIR/f"XGB_VIF{t}.joblib"
 if f.exists(): model=joblib.load(f)
 else:
  model=Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("model",XGBClassifier(random_state=RANDOM_STATE,eval_metric="logloss",n_jobs=-1,**clean))]); model.fit(tr[keep],tr[TARGET].astype(int)); joblib.dump(model,f)
 pr=model.predict_proba(te[keep])[:,1]; q=(pr>=.5).astype(int); y=te[TARGET].astype(int)
 rows.append({"VIF_threshold":t,"n_predictors":len(keep),"retained_predictors":";".join(keep),"AUC":roc_auc_score(y,pr),"Macro_F1":f1_score(y,q,average="macro"),"MCC":matthews_corrcoef(y,q)})
pd.concat(hs,ignore_index=True).to_csv(VIF_DIR/"VIF_elimination_history.csv",index=False); o=pd.DataFrame(rows); o.to_csv(VIF_DIR/"VIF_sensitivity_metrics.csv",index=False); print(o.round(4).to_string(index=False))
