"""Shared configuration for Appendix A integrated fire severity modelling."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Keep input datasets and generated outputs outside the public repository.
DATA_DIR = Path(os.environ.get("APPENDIX_DATA_DIR", str(ROOT / "private_data"))).expanduser()
OUTPUT_ROOT = Path(os.environ.get("APPENDIX_OUTPUT_DIR", str(ROOT / "private_outputs"))).expanduser()
DATA_FILE = DATA_DIR / os.environ.get("APPENDIX_DATA_FILE", "main_modelling_data.csv")
SPATIAL_DATA_FILE = DATA_DIR / os.environ.get("APPENDIX_SPATIAL_DATA_FILE", "spatial_modelling_data.csv")

VAL_DIR = OUTPUT_ROOT / "appendix_validation_outputs"
ADD_DIR = OUTPUT_ROOT / "appendix_additional_experiments"
FIG_DIR = OUTPUT_ROOT / "appendix_final_figures"
SHAP_DIR = OUTPUT_ROOT / "appendix_shap_outputs"
GAM_DIR = OUTPUT_ROOT / "appendix_gam_outputs"
VIF_DIR = OUTPUT_ROOT / "appendix_vif_outputs"
SPATIAL_DIR = OUTPUT_ROOT / "appendix_spatial_outputs"
HETEROGENEITY_DIR = OUTPUT_ROOT / "appendix_spatial_heterogeneity"
BLOCK_DIR = OUTPUT_ROOT / "appendix_spatial_block_validation"
DOMAIN_CONTEXT_DIR = OUTPUT_ROOT / "appendix_domain_context"

for folder in (VAL_DIR, ADD_DIR, FIG_DIR, SHAP_DIR, GAM_DIR, VIF_DIR,
               SPATIAL_DIR, HETEROGENEITY_DIR, BLOCK_DIR, DOMAIN_CONTEXT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

ID_COL = "pt_id"
# Binary Low/High fire-severity response derived from RdNBR.
# The historical field name is retained for compatibility with the modelling dataset.
TARGET = "SNIC2clsN"
CONTINUOUS_SEVERITY = "RdNBR"
LANDFORM_COL = "LF4Cls"
VEGETATION_COL = "Veg5Cls"
X_COL = "x_cood"
Y_COL = "y_cood"
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
FIGURE_DPI = 400

METEOROLOGY = ["avgMaxTempSn","avgSurfPresSn","TotRainSn","TotRainYr","avgRHtmaxSn","avgV10Sn","avgU10Sn","avgAbsWndSn","avgSwvL1Sn","SoilMois_0to10"]
TOPOGRAPHY = ["DEM","SlpPerc","AspCosine","TWI","SolDur","SolDifRad","ConvIndx","ProfCurv","PlanCurv","LocCurv","DwnSlpCurv","UpSlpCurv"]
FUEL_VEGETATION = ["NDMI_mean","LAI","LFMC","EVI","DFFM_silo","FCOV_BS","FCOV_PV","Flamm","CHM"]
ALL_PREDICTORS = METEOROLOGY + TOPOGRAPHY + FUEL_VEGETATION
BASE_MODELS = ["RF","XGB","SVM","MLP"]

DOMAIN_FEATURES = {
    "M": METEOROLOGY,
    "T": TOPOGRAPHY,
    "FV": FUEL_VEGETATION,
    "M+T": METEOROLOGY + TOPOGRAPHY,
    "M+FV": METEOROLOGY + FUEL_VEGETATION,
    "T+FV": TOPOGRAPHY + FUEL_VEGETATION,
    "M+T+FV": ALL_PREDICTORS,
}
LANDFORM_ORDER = ["Peak", "UpSlp", "LwSlp", "Valy"]
VEGETATION_ORDER = ["RFVT", "EFWL", "OFWL", "SHMW"]
XGB_MODEL_FILE = VAL_DIR / "base_XGB_final.joblib"
