from pathlib import Path

# ========================================
# Recovery analysis configuration files.. 
# ========================================


import os

# Configure private directories outside this repository using environment variables.
# RECOVERY_DATA_DIR points to the folder containing 02_Long_Format/.
# RECOVERY_OUTPUT_DIR points to the analysis output root.
def required_directory(name):
    value = os.environ.get(name)
    if not value:
        raise EnvironmentError(
            f"Set the {name} environment variable to your private local directory "
            "before running the recovery scripts. See README.md."
        )
    return Path(value).expanduser()

DATA_ROOT = required_directory("RECOVERY_DATA_DIR")
PREFERRED_INPUT_LONG = DATA_ROOT / "02_Long_Format" / "Recovery_Climate_Long.csv"


def locate_long_csv():
    if PREFERRED_INPUT_LONG.exists():
        return PREFERRED_INPUT_LONG
    candidates = sorted(DATA_ROOT.rglob("*Long*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "Long-format recovery CSV not found under RECOVERY_DATA_DIR. "
            "See README.md for the expected directory layout."
        )
    raise RuntimeError(
        "Multiple long-format CSV candidates found. Place the intended input "
        "at 02_Long_Format/Recovery_Climate_Long.csv."
    )


INPUT_LONG = locate_long_csv()
ANALYSIS_ROOT = required_directory("RECOVERY_OUTPUT_DIR")
DATA_DIR = ANALYSIS_ROOT / "01_Analysis_Ready"
MODEL_DIR = ANALYSIS_ROOT / "02_Model_Results"
FIGURE_DIR = ANALYSIS_ROOT / "03_Final_Figures"
TABLE_DIR = ANALYSIS_ROOT / "04_Final_Tables"
SUPP_DIR = ANALYSIS_ROOT / "05_Supplementary"
LOG_DIR = ANALYSIS_ROOT / "06_Run_Logs"

RECOVERY_DIR = MODEL_DIR / "Recovery"
CHM_DIR = MODEL_DIR / "CHM"
CLIMATE_DIR = MODEL_DIR / "Climate"
SUPP_RECOVERY_DIR = SUPP_DIR / "Recovery"
SUPP_CHM_DIR = SUPP_DIR / "CHM"
SUPP_CLIMATE_DIR = SUPP_DIR / "Climate"
SUPP_INDEX_DIR = SUPP_DIR / "Supporting_Indices"

for folder in [
    ANALYSIS_ROOT, DATA_DIR, MODEL_DIR, FIGURE_DIR, TABLE_DIR, SUPP_DIR, LOG_DIR,
    RECOVERY_DIR, CHM_DIR, CLIMATE_DIR,
    SUPP_RECOVERY_DIR, SUPP_CHM_DIR, SUPP_CLIMATE_DIR, SUPP_INDEX_DIR,
]:
    folder.mkdir(parents=True, exist_ok=True)

ANALYSIS_READY = DATA_DIR / "Recovery_Analysis_Ready.csv"

TIMES = ["T1", "T2", "T3", "T4", "T5"]
SEVERITIES = ["Low", "High"]
LANDFORMS = ["Peak", "UpSlp", "LwSlp", "PlnValy"]
VEGETATION = ["EFWL", "RFVT", "OFWL", "SHMW"]
INTEGRATED_VEGETATION = ["EFWL", "SHMW"]
PRIMARY_VEGETATION = "EFWL"

LANDFORM_LABELS = {
    "Peak": "Peak",
    "UpSlp": "UpSlp",
    "LwSlp": "LwSlp",
    "PlnValy": "PlnValy",
}

VEGETATION_LABELS = {
    "EFWL": "EFWL",
    "RFVT": "RFVT",
    "OFWL": "OFWL",
    "SHMW": "SHMW",
}

LOW_COLOUR = "#0072B2"
HIGH_COLOUR = "#D55E00"
LANDFORM_COLOURS = {
    "Peak": "#0072B2",
    "UpSlp": "#009E73",
    "LwSlp": "#E69F00",
    "PlnValy": "#CC79A7",
}

FIG_DPI = 600
RUN_GAM = True
RECOVERY_THRESHOLDS = [0.80, 1.00]
