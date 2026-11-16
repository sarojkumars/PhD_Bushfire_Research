"""Appendix A Script 16: environmental-domain predictive information by context.

The analysis quantifies predictive information, not causal effects.
"""
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from xgboost import XGBClassifier
from appendix_config import *

DOMAIN_CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
OUT = DOMAIN_CONTEXT_DIR
PRED_DIR = OUT / "predictions"; PRED_DIR.mkdir(exist_ok=True)
MODEL_DIR = OUT / "models"; MODEL_DIR.mkdir(exist_ok=True)
DOMAIN_ORDER = ["M","T","FV","M+T","M+FV","T+FV","M+T+FV"]
MIN_TEST_N = 100
MIN_CLASS_N = 20

INCREMENTS = [
    ("Add T to M","M","M+T"), ("Add FV to M","M","M+FV"),
    ("Add M to T","T","M+T"), ("Add FV to T","T","T+FV"),
    ("Add M to FV","FV","M+FV"), ("Add T to FV","FV","T+FV"),
    ("Add FV to M+T","M+T","M+T+FV"),
    ("Add T to M+FV","M+FV","M+T+FV"),
    ("Add M to T+FV","T+FV","M+T+FV"),
]

def extract_xgb(obj):
    if isinstance(obj, XGBClassifier): return obj
    if hasattr(obj, "named_steps"):
        for v in obj.named_steps.values():
            if isinstance(v, XGBClassifier): return v
    if isinstance(obj, dict):
        for v in obj.values():
            if isinstance(v, XGBClassifier): return v
            if hasattr(v, "named_steps"):
                for z in v.named_steps.values():
                    if isinstance(z, XGBClassifier): return z
    return None

def reference_params():
    if not XGB_MODEL_FILE.exists():
        raise FileNotFoundError(f"Run Script 02 first. Final XGB model not found: {XGB_MODEL_FILE}")
    est = extract_xgb(joblib.load(XGB_MODEL_FILE))
    if est is None: raise TypeError(f"Could not extract XGBClassifier from {XGB_MODEL_FILE}")
    p = est.get_params()
    keep = ["n_estimators","max_depth","max_leaves","learning_rate","subsample","colsample_bytree",
            "colsample_bylevel","colsample_bynode","min_child_weight","gamma","reg_alpha","reg_lambda",
            "grow_policy","max_bin","scale_pos_weight","booster"]
    out = {k:p[k] for k in keep if k in p and p[k] is not None}
    (OUT/"xgb_parameters_used.json").write_text(json.dumps(out,indent=2,default=str),encoding="utf-8")
    return out

def load_data():
    if not DATA_FILE.exists(): raise FileNotFoundError(f"Main data not found: {DATA_FILE}")
    sf = VAL_DIR / "01_fixed_split_ids.csv"
    if not sf.exists(): raise FileNotFoundError(f"Run Script 01/02 first. Split file not found: {sf}")
    df = pd.read_csv(DATA_FILE)
    sp = pd.read_csv(sf)
    req = [ID_COL,TARGET,LANDFORM_COL,VEGETATION_COL]+ALL_PREDICTORS
    miss=[c for c in req if c not in df.columns]
    if miss: raise ValueError(f"Missing required columns: {miss}")
    if df[ID_COL].duplicated().any(): raise ValueError(f"Duplicate {ID_COL} values in modelling data")
    spids=set(sp[ID_COL]); dfids=set(df[ID_COL])
    if not spids.issubset(dfids): raise ValueError("Fixed split contains IDs absent from DATA_FILE; do not mix datasets/splits.")
    m=df.merge(sp[[ID_COL,"split"]],on=ID_COL,how="inner",validate="one_to_one")
    tr=m[m.split=="TRAIN"].copy(); te=m[m.split=="TEST"].copy()
    if len(tr)+len(te)!=len(sp): raise ValueError("Split/data merge did not reproduce all fixed-split observations")
    tr[TARGET]=tr[TARGET].astype(int); te[TARGET]=te[TARGET].astype(int)
    return tr,te

def metrics(y,p):
    yh=(p>=.5).astype(int)
    return dict(
        auc=roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,
        accuracy=accuracy_score(y,yh), precision=precision_score(y,yh,zero_division=0),
        recall=recall_score(y,yh,zero_division=0), f1=f1_score(y,yh,zero_division=0),
        macro_f1=f1_score(y,yh,average="macro",zero_division=0), mcc=matthews_corrcoef(y,yh))

def fit_domain_models(tr,te,pars):
    preds={}
    for d in DOMAIN_ORDER:
        feats=DOMAIN_FEATURES[d]
        model=Pipeline([("imputer",SimpleImputer(strategy="median")),
                        ("model",XGBClassifier(**pars,objective="binary:logistic",eval_metric="logloss",
                                               random_state=RANDOM_STATE,n_jobs=-1,tree_method="hist"))])
        print(f"Fitting {d:7s} ({len(feats)} predictors) ...")
        model.fit(tr[feats],tr[TARGET])
        joblib.dump(model,MODEL_DIR/f"XGB_domain_{d.replace('+','_')}.joblib")
        preds[d]=model.predict_proba(te[feats])[:,1]
    base=te[[ID_COL,TARGET,LANDFORM_COL,VEGETATION_COL]].reset_index(drop=True).copy()
    for d,p in preds.items(): base[f"prob_{d}"]=p
    base.to_csv(OUT/"domain_context_test_predictions.csv",index=False)
    return base

def context_labels(series, preferred):
    observed=[str(x) for x in series.dropna().unique()]
    return [x for x in preferred if x in observed]

def evaluate_contexts(pred):
    rows=[]; skipped=[]
    families=[("landform",LANDFORM_COL,LANDFORM_ORDER),("vegetation",VEGETATION_COL,VEGETATION_ORDER)]
    for typ,col,order in families:
        for level in context_labels(pred[col],order):
            q=pred[pred[col].astype(str)==level].copy(); y=q[TARGET].to_numpy(int)
            n0=int((y==0).sum()); n1=int((y==1).sum())
            if len(q)<MIN_TEST_N or min(n0,n1)<MIN_CLASS_N:
                skipped.append(dict(context_type=typ,context_level=level,n=len(q),n_low=n0,n_high=n1,
                                    reason=f"requires n>={MIN_TEST_N} and each class>={MIN_CLASS_N}")); continue
            for d in DOMAIN_ORDER:
                r=dict(context_type=typ,context_level=level,domain_set=d,n=len(q),n_low=n0,n_high=n1,
                       prevalence_high=float(y.mean()))
                r.update(metrics(y,q[f"prob_{d}"].to_numpy(float))); rows.append(r)
    return pd.DataFrame(rows),pd.DataFrame(skipped)

def increments(tab):
    rows=[]
    for (typ,lev),g in tab.groupby(["context_type","context_level"]):
        z=g.set_index("domain_set")
        for label,a,b in INCREMENTS:
            if a in z.index and b in z.index:
                rows.append(dict(context_type=typ,context_level=lev,comparison=label,baseline=a,expanded=b,
                    n=int(z.loc[b,"n"]),baseline_auc=z.loc[a,"auc"],expanded_auc=z.loc[b,"auc"],delta_auc=z.loc[b,"auc"]-z.loc[a,"auc"],
                    delta_macro_f1=z.loc[b,"macro_f1"]-z.loc[a,"macro_f1"],delta_mcc=z.loc[b,"mcc"]-z.loc[a,"mcc"]))
    return pd.DataFrame(rows)

def plot_heatmap(tab,typ):
    q=tab[tab.context_type==typ]
    if q.empty:return
    p=q.pivot(index="context_level",columns="domain_set",values="auc").reindex(columns=DOMAIN_ORDER)
    fig,ax=plt.subplots(figsize=(9,max(3.5,.7*len(p))))
    im=ax.imshow(p.values,aspect="auto")
    ax.set_xticks(range(len(p.columns)),p.columns); ax.set_yticks(range(len(p.index)),p.index)
    ax.set_xlabel("Environmental predictor information"); ax.set_ylabel(typ.capitalize())
    ax.set_title(f"Environmental-domain discrimination across {typ} contexts")
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if np.isfinite(p.iloc[i,j]): ax.text(j,i,f"{p.iloc[i,j]:.3f}",ha="center",va="center")
    fig.colorbar(im,ax=ax,label="AUC"); fig.tight_layout()
    fig.savefig(OUT/f"Fig_domain_AUC_{typ}.png",dpi=FIGURE_DPI,bbox_inches="tight"); plt.close(fig)

def main():
    print("="*78); print("SCRIPT 16 - ENVIRONMENTAL-DOMAIN INFORMATION BY CONTEXT"); print("="*78)
    print("Main data:",DATA_FILE); print("Fixed split:",VAL_DIR/"01_fixed_split_ids.csv")
    tr,te=load_data(); print(f"Train={len(tr):,}; test={len(te):,}")
    pars=reference_params(); pred=fit_domain_models(tr,te,pars)
    tab,skip=evaluate_contexts(pred); inc=increments(tab)
    tab.to_csv(OUT/"domain_context_model_performance.csv",index=False)
    inc.to_csv(OUT/"domain_context_incremental_auc.csv",index=False)
    if not skip.empty: skip.to_csv(OUT/"domain_context_skipped_groups.csv",index=False)
    for typ in ["landform","vegetation"]:
        plot_heatmap(tab,typ)
        q=tab[tab.context_type==typ].pivot(index="context_level",columns="domain_set",values="auc").reindex(columns=DOMAIN_ORDER)
        q.to_csv(OUT/f"{typ}_domain_auc_matrix.csv")
        qi=inc[inc.context_type==typ].pivot(index="context_level",columns="comparison",values="delta_auc")
        qi.to_csv(OUT/f"{typ}_incremental_auc_matrix.csv")
    with open(OUT/"domain_context_summary.txt","w",encoding="utf-8") as f:
        f.write("Context-specific environmental-domain predictive information\n")
        f.write("Predictive associations only; not causal effects.\n\n")
        f.write(tab.round(4).to_string(index=False)); f.write("\n\nIncremental information\n")
        f.write(inc.round(4).to_string(index=False))
        if not skip.empty: f.write("\n\nSkipped small groups\n"+skip.to_string(index=False))
    print("\nSaved:",OUT); print(tab.round(4).to_string(index=False)); print("\nIncremental AUC:\n",inc.round(4).to_string(index=False))

if __name__ == "__main__": main()
