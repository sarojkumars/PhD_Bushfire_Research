import pandas as pd
from appendix_config import *
w=pd.read_csv(VAL_DIR/"attention_weights_FC.csv"); rows=[]
for typ,col in [("landform",LANDFORM_COL),("vegetation",VEGETATION_COL)]:
 for level,g in w.groupby(col,dropna=False):
  for m in BASE_MODELS: rows.append(dict(context_type=typ,context_level=level,base_model=m,mean_attention_weight=g[m].mean(),sd_attention_weight=g[m].std(),n=len(g)))
o=pd.DataFrame(rows); o.to_csv(VAL_DIR/"08_FC_attention_weights_by_context_FINAL.csv",index=False); print(o.round(4).to_string(index=False))
