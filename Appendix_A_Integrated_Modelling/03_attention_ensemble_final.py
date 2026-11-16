from pathlib import Path
import runpy
from appendix_config import VAL_DIR
needed=[VAL_DIR/"05_context_ablation_metrics.csv",VAL_DIR/"06_context_ablation_predictions.csv",VAL_DIR/"attention_FC.keras"]
if all(p.exists() for p in needed): print("Ensemble outputs already exist; reused.")
else: runpy.run_path(str(Path(__file__).with_name("02_base_models_and_main_validation.py")),run_name="__main__")
