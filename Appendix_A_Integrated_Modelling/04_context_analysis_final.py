import numpy as np,pandas as pd
from sklearn.metrics import *
from appendix_config import *
p=pd.read_csv(VAL_DIR/"06_context_ablation_predictions.csv"); rows=[]
for variant in ["CF","TC","VC","FC"]:
 for typ,col in [("landform",LANDFORM_COL),("vegetation",VEGETATION_COL)]:
  for level,g in p.groupby(col,dropna=False):
   y=g[TARGET].astype(int).to_numpy(); pr=g[f"prob_{variant}"].to_numpy(); pred=(pr>=.5).astype(int)
   rows.append(dict(variant=variant,context_type=typ,context_level=level,n=len(g),prevalence=y.mean(),
    auc=roc_auc_score(y,pr) if len(np.unique(y))==2 else np.nan,accuracy=accuracy_score(y,pred),
    f1_class1=f1_score(y,pred,zero_division=0),f1_macro=f1_score(y,pred,average="macro",zero_division=0),
    f1_weighted=f1_score(y,pred,average="weighted",zero_division=0),mcc=matthews_corrcoef(y,pred),
    precision_class1=precision_score(y,pred,zero_division=0),recall_class1=recall_score(y,pred,zero_division=0)))
o=pd.DataFrame(rows); o.to_csv(VAL_DIR/"07_context_specific_test_metrics_RECALCULATED.csv",index=False)
print(o.round(4).to_string(index=False))
