import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,f1_score,matthews_corrcoef
from appendix_config import *
N_BOOT=2000; rng=np.random.default_rng(RANDOM_STATE)
ctx=pd.read_csv(VAL_DIR/"06_context_ablation_predictions.csv"); base=pd.read_csv(VAL_DIR/"04_test_base_probabilities.csv")
if len(base)!=len(ctx): raise ValueError("Saved probability lengths differ.")
y=ctx[TARGET].astype(int).to_numpy()
cmp={"FC - CF":(ctx.prob_CF.values,ctx.prob_FC.values),"TC - CF":(ctx.prob_CF.values,ctx.prob_TC.values),
"VC - CF":(ctx.prob_CF.values,ctx.prob_VC.values),"CF ensemble - XGB":(base.XGB.values,ctx.prob_CF.values),
"FC ensemble - XGB":(base.XGB.values,ctx.prob_FC.values)}
def met(yy,p):
 q=(p>=.5).astype(int); return np.array([roc_auc_score(yy,p),f1_score(yy,q,average="macro"),matthews_corrcoef(yy,q)])
rows=[]; n=len(y)
for name,(a,b) in cmp.items():
 point=met(y,b)-met(y,a); boot=[]
 for _ in range(N_BOOT):
  ix=rng.integers(0,n,n); yy=y[ix]
  if len(np.unique(yy))==2: boot.append(met(yy,b[ix])-met(yy,a[ix]))
 boot=np.asarray(boot)
 for j,m in enumerate(["AUC","Macro-F1","MCC"]):
  lo,hi=np.quantile(boot[:,j],[.025,.975]); rows.append(dict(comparison=name,metric=m,difference=point[j],ci_low=lo,ci_high=hi,n_boot=len(boot)))
o=pd.DataFrame(rows); o.to_csv(VAL_DIR/"13_paired_bootstrap_differences.csv",index=False); print(o.round(6).to_string(index=False))
