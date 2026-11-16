from pathlib import Path
import runpy
from appendix_config import VAL_DIR
needed=[VAL_DIR/"10_environmental_domain_ablation_metrics.csv",VAL_DIR/"11_environmental_domain_ablation_predictions.csv",VAL_DIR/"domain_ablation_XGB_M_T_FV.joblib"]
if all(p.exists() for p in needed): print("Domain-ablation joblib/results already exist; reused.")
else: runpy.run_path(str(Path(__file__).with_name("02_base_models_and_main_validation.py")),run_name="__main__")
