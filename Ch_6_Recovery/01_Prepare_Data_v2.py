import importlib.util
from pathlib import Path
import pandas as pd

spec = importlib.util.spec_from_file_location("config", Path(__file__).with_name("00_Config_v2.py"))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)

ALIASES = {
    "Sample_ID": ["Sample_ID", "SampleID", "ID", "Point_ID"],
    "Time": ["Time", "Recovery_Time", "Stage"],
    "Recovery": ["Recovery", "NBR_Recovery", "NBRRecovery", "NBR_Recov", "Recov_NBR", "NBRRecov"],
    "Severity": ["Severity", "Sev2Class", "Severity_Class", "Severity2Cls"],
    "Landform": ["Landform", "Landform_Label", "Topo4Cls", "Landform4Cls"],
    "Vegetation": ["Vegetation", "Veg4Cls", "Vegetation_Group", "Veg_Group"],
    "CHM": ["CHM", "CanopyHeight", "Canopy_Height"],
    "RfSpr": ["RfSpr", "RainfallSpr", "RainSpr"],
    "SMSpr": ["SMSpr", "SMSpring", "SoilMoistureSpr"],
    "TMaxSpr": ["TMaxSpr", "TmaxSpr", "MaxTempSpr"],
}


def apply_aliases(df):
    rename = {}
    for target, candidates in ALIASES.items():
        if target in df.columns:
            continue
        for candidate in candidates:
            if candidate in df.columns:
                rename[candidate] = target
                break
    return df.rename(columns=rename)


def require(df, cols):
    missing = [x for x in cols if x not in df.columns]
    if missing:
        raise ValueError(
            "Missing required fields:\n  - " + "\n  - ".join(missing)
            + "\n\nEdit ALIASES in 01_Prepare_Data_v2.py if your column names differ."
        )


df = apply_aliases(pd.read_csv(c.INPUT_LONG))
require(df, [
    "Sample_ID", "Time", "Recovery", "Severity", "Landform", "Vegetation",
    "CHM", "RfSpr", "SMSpr", "TMaxSpr"
])

for col in ["Time", "Severity", "Landform", "Vegetation"]:
    df[col] = df[col].astype("string").str.strip()

df["Landform"] = df["Landform"].replace({
    "Valy": "PlnValy",
    "Plain/Valley": "PlnValy",
    "Plain-Valley": "PlnValy",
    "Upper slope": "UpSlp",
    "Upper Slope": "UpSlp",
    "Lower slope": "LwSlp",
    "Lower Slope": "LwSlp",
})

df = df[df["Time"].isin(c.TIMES)].copy()
df = df[df["Severity"].isin(c.SEVERITIES)].copy()
df = df[df["Landform"].isin(c.LANDFORMS)].copy()

for col in ["Recovery", "CHM", "RfSpr", "SMSpr", "TMaxSpr"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Time"] = pd.Categorical(df["Time"], categories=c.TIMES, ordered=True)
df["Severity"] = pd.Categorical(df["Severity"], categories=c.SEVERITIES, ordered=True)
df["Landform"] = pd.Categorical(df["Landform"], categories=c.LANDFORMS, ordered=True)

n_dup = int(df.duplicated(["Sample_ID", "Time"]).sum())
if n_dup:
    raise ValueError(f"Found {n_dup:,} duplicate Sample_ID × Time records. Resolve duplicates before modelling.")

df.to_csv(c.ANALYSIS_READY, index=False)

unique = df.drop_duplicates("Sample_ID")
audit = pd.DataFrame({
    "Metric": ["Rows", "Unique Sample_ID", "Low severity IDs", "High severity IDs", "EFWL IDs", "RFVT IDs", "OFWL IDs", "SHMW IDs"],
    "Value": [
        len(df), unique["Sample_ID"].nunique(),
        (unique["Severity"] == "Low").sum(), (unique["Severity"] == "High").sum(),
        (unique["Vegetation"] == "EFWL").sum(), (unique["Vegetation"] == "RFVT").sum(),
        (unique["Vegetation"] == "OFWL").sum(), (unique["Vegetation"] == "SHMW").sum(),
    ],
})
audit.to_csv(c.TABLE_DIR / "Table_Data_Audit.csv", index=False)

missing = (
    df[
        [
            "Recovery", "Severity", "Landform", "Vegetation",
            "CHM", "RfSpr", "SMSpr", "TMaxSpr"
        ]
    ]
    .isna()
    .sum()
    .rename("Missing_n")
    .reset_index()
)

missing.columns = ["Variable", "Missing_n"]
missing.to_csv(c.SUPP_RECOVERY_DIR / "S_Data_Missingness.csv", index=False)

print("=" * 72)
print("RECOVERY ANALYSIS V2 — DATA PREPARATION COMPLETE")
print("=" * 72)
print(f"Input : {c.INPUT_LONG}")
print(f"Output: {c.ANALYSIS_READY}")
print(f"Rows  : {len(df):,}")
print(f"IDs   : {df['Sample_ID'].nunique():,}")
print(f"Root  : {c.ANALYSIS_ROOT}")
