import pandas as pd
from sklearn.model_selection import train_test_split
from appendix_config import *
VAL_DIR.mkdir(exist_ok=True)
if not DATA_FILE.exists(): raise FileNotFoundError(f"Edit DATA_FILE in appendix_config.py: {DATA_FILE}")
df=pd.read_csv(DATA_FILE)
req=[ID_COL,TARGET,LANDFORM_COL,VEGETATION_COL]+ALL_PREDICTORS
miss=[c for c in req if c not in df.columns]
if miss: raise ValueError(f"Missing columns: {miss}")
df=df.dropna(subset=[ID_COL,TARGET]).copy(); df[TARGET]=df[TARGET].astype(int)
if df[ID_COL].duplicated().any(): raise ValueError("Duplicate pt_id values found.")
split_file=VAL_DIR/"01_fixed_split_ids.csv"
if split_file.exists():
    split=pd.read_csv(split_file)
    if set(split[ID_COL])!=set(df[ID_COL]): raise ValueError("Existing split does not match current dataset.")
    print("Reusing fixed split.")
else:
    tr,te=train_test_split(df[[ID_COL,TARGET]],test_size=.2,stratify=df[TARGET],random_state=RANDOM_STATE)
    split=pd.concat([tr.assign(split="TRAIN"),te.assign(split="TEST")],ignore_index=True)
    split.to_csv(split_file,index=False)
summary=pd.DataFrame([{"n_total":len(df),"n_train":(split.split=="TRAIN").sum(),"n_test":(split.split=="TEST").sum(),
 "target_prevalence":df[TARGET].mean(),"missing_vegetation_n":df[VEGETATION_COL].isna().sum(),
 "missing_vegetation_pct":100*df[VEGETATION_COL].isna().mean()}])
summary.to_csv(VAL_DIR/"00_data_summary.csv",index=False)
v=df[VEGETATION_COL].fillna("<MISSING>").value_counts().rename_axis("vegetation").reset_index(name="n")
v["pct"]=100*v.n/len(df); v.to_csv(VAL_DIR/"00_vegetation_context_summary.csv",index=False)
print(summary.to_string(index=False)); print(v.to_string(index=False))
