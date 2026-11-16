# Appendix A — SHAP for the full-context attention ensemble
import json
import os
import random
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from appendix_config import DATA_FILE, VAL_DIR, ID_COL, TARGET, LANDFORM_COL, VEGETATION_COL, RANDOM_STATE, BASE_MODELS
# Settings
SCRIPT_DIR = Path(__file__).resolve().parent
ENSEMBLE_DIR = SCRIPT_DIR / 'appendix_innovative_ensemble'
OUT_DIR = SCRIPT_DIR / 'appendix_ensemble_shap_1500'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_FILE = VAL_DIR / '01_fixed_split_ids.csv'
FC_MODEL_FILE = ENSEMBLE_DIR / 'final_FC_original_attention.keras'
CONTEXT_COLUMNS_FILE = ENSEMBLE_DIR / '00_context_columns.json'
N_EXPLAIN = 1500
N_BACKGROUND = 50
N_PERMUTATION_CYCLES = 3
PREDICTION_BATCH_SIZE = 512
MIN_CONTEXT_N = 20
TOP_N = 20
METEOROLOGY = ['avgMaxTempSn', 'avgSurfPresSn', 'TotRainSn', 'TotRainYr', 'avgRHtmaxSn', 'avgV10Sn', 'avgU10Sn', 'avgAbsWndSn', 'avgSwvL1Sn', 'SoilMois_0to10']
TOPOGRAPHY = ['DEM', 'SlpPerc', 'AspCosine', 'TWI', 'SolDur', 'SolDifRad', 'ConvIndx', 'ProfCurv', 'PlanCurv', 'LocCurv', 'DwnSlpCurv', 'UpSlpCurv']
FUEL_VEGETATION = ['NDMI_mean', 'LAI', 'LFMC', 'EVI', 'DFFM_silo', 'FCOV_BS', 'FCOV_PV', 'Flamm', 'CHM']
ALL_PREDICTORS = METEOROLOGY + TOPOGRAPHY + FUEL_VEGETATION
DOMAIN_MAP = {**{feature: 'M' for feature in METEOROLOGY}, **{feature: 'T' for feature in TOPOGRAPHY}, **{feature: 'FV' for feature in FUEL_VEGETATION}}

# Random seeds
def set_all_seeds(seed: int=RANDOM_STATE) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass
set_all_seeds()
# Load models and data
required_files = [DATA_FILE, SPLIT_FILE, FC_MODEL_FILE, CONTEXT_COLUMNS_FILE]
for model_name in BASE_MODELS:
    required_files.append(VAL_DIR / f'base_{model_name}_final.joblib')
missing_files = [str(path) for path in required_files if not Path(path).exists()]
if missing_files:
    raise FileNotFoundError('\nMissing required files:\n\n' + '\n'.join(missing_files))
print('\n' + '=' * 78)
print('DIRECT ENSEMBLE-LEVEL SHAP')
print('=' * 78)
data = pd.read_csv(DATA_FILE)
split = pd.read_csv(SPLIT_FILE)
required_columns = [ID_COL, TARGET, LANDFORM_COL, VEGETATION_COL] + ALL_PREDICTORS
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError('Missing required columns in modelling data:\n' + '\n'.join(missing_columns))
train_ids = split.loc[split['split'] == 'TRAIN', ID_COL].tolist()
test_ids = split.loc[split['split'] == 'TEST', ID_COL].tolist()
indexed_data = data.set_index(ID_COL)
train_df = indexed_data.loc[train_ids].reset_index()
test_df = indexed_data.loc[test_ids].reset_index()
print(f'\nTraining observations: {len(train_df):,}')
print(f'Held-out test observations: {len(test_df):,}')
print(f'Environmental predictors: {len(ALL_PREDICTORS)}')
base_models = {}
for model_name in BASE_MODELS:
    model_file = VAL_DIR / f'base_{model_name}_final.joblib'
    print(f'Loading final base model: {model_name}')
    base_models[model_name] = joblib.load(model_file)
print('\nLoaded base models:', list(base_models.keys()))
print('\nLoading final FC attention model:')
print(FC_MODEL_FILE)
fc_model = tf.keras.models.load_model(FC_MODEL_FILE, compile=False, safe_mode=False)
print('\nFC model input shapes:')
for inp in fc_model.inputs:
    print(f'  {inp.name}: {inp.shape}')
print(f'FC model output shape: {fc_model.output_shape}')
with open(CONTEXT_COLUMNS_FILE, 'r', encoding='utf-8') as f:
    context_names = json.load(f)
full_context_names = context_names['full']
print(f'\nFull context variables: {len(full_context_names)}')

# Context encoding
def make_context_matrix(df: pd.DataFrame) -> np.ndarray:
    lf = df[LANDFORM_COL].fillna('Missing').astype(str)
    vg = df[VEGETATION_COL].fillna('Missing').astype(str)
    lf_dummy = pd.get_dummies(lf, prefix='topo', dtype=float)
    vg_dummy = pd.get_dummies(vg, prefix='veg', dtype=float)
    ctx = pd.concat([lf_dummy, vg_dummy], axis=1)
    ctx = ctx.reindex(columns=full_context_names, fill_value=0.0)
    return ctx.to_numpy(dtype=np.float32)
train_context = make_context_matrix(train_df)
test_context = make_context_matrix(test_df)
expected_context_width = int(fc_model.inputs[1].shape[-1])
if test_context.shape[1] != expected_context_width:
    raise RuntimeError(f'Context width does not match final FC model.\nContext matrix: {test_context.shape[1]}\nFC model expects: {expected_context_width}')

# Ensemble predictions
def base_probability_matrix(X) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        X_df = X[ALL_PREDICTORS].copy()
    else:
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2 or X_array.shape[1] != len(ALL_PREDICTORS):
            raise ValueError(f'Environmental predictor matrix has shape {X_array.shape}; expected (n, {len(ALL_PREDICTORS)}).')
        X_df = pd.DataFrame(X_array, columns=ALL_PREDICTORS)
    probability_blocks = []
    for model_name in BASE_MODELS:
        probs = base_models[model_name].predict_proba(X_df)
        probs = np.asarray(probs, dtype=np.float32)
        if probs.ndim != 2 or probs.shape[1] != 2:
            raise RuntimeError(f'{model_name} returned unexpected probability shape {probs.shape}.')
        probability_blocks.append(probs)
    return np.hstack(probability_blocks).astype(np.float32)

def ensemble_predict_with_context(X, context_matrix) -> np.ndarray:
    X_array = np.asarray(X, dtype=float)
    context_matrix = np.asarray(context_matrix, dtype=np.float32)
    if X_array.shape[0] != context_matrix.shape[0]:
        raise ValueError('Number of environmental observations and context rows must match.')
    n = X_array.shape[0]
    outputs = []
    for start in range(0, n, PREDICTION_BATCH_SIZE):
        end = min(start + PREDICTION_BATCH_SIZE, n)
        X_batch = X_array[start:end]
        context_batch = context_matrix[start:end]
        prob8 = base_probability_matrix(X_batch)
        pred2 = fc_model.predict([prob8, context_batch], verbose=0)
        pred2 = np.asarray(pred2)
        if pred2.ndim != 2 or pred2.shape[1] != 2:
            raise RuntimeError(f'Final FC model returned unexpected shape {pred2.shape}; expected (n, 2).')
        outputs.append(pred2[:, 1])
    return np.concatenate(outputs)
print('\nVerifying reconstructed full ensemble...')
full_test_probability = ensemble_predict_with_context(test_df[ALL_PREDICTORS].to_numpy(), test_context)
test_auc = roc_auc_score(test_df[TARGET].astype(int), full_test_probability)
print(f'Reconstructed FC test AUC: {test_auc:.6f}')
EXPECTED_AUC = 0.94015
AUC_TOLERANCE = 0.002
if abs(test_auc - EXPECTED_AUC) > AUC_TOLERANCE:
    raise RuntimeError(f'\nSTOPPING BEFORE SHAP.\n\nThe reconstructed FC ensemble does not reproduce the expected final test performance.\nObserved AUC: {test_auc:.6f}\nExpected approximately: {EXPECTED_AUC:.6f}\n\nCheck that the final base-model joblib files and final_FC_original_attention.keras belong to the same final modelling run.')
print('Full ensemble verification passed.')
# SHAP sample
test_sampling = test_df.copy()
test_sampling['_landform_str'] = test_sampling[LANDFORM_COL].fillna('Missing').astype(str)
test_sampling['_vegetation_str'] = test_sampling[VEGETATION_COL].fillna('Missing').astype(str)
test_sampling['_stratum'] = test_sampling[TARGET].astype(str) + '__' + test_sampling['_landform_str'] + '__' + test_sampling['_vegetation_str']
rng = np.random.default_rng(RANDOM_STATE)

def stratified_sample(df: pd.DataFrame, n: int, stratum_col: str, seed: int) -> pd.DataFrame:
    if n >= len(df):
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    groups = df.groupby(stratum_col, dropna=False)
    allocations = {}
    for name, group in groups:
        proportional_n = int(round(n * len(group) / len(df)))
        allocations[name] = min(len(group), max(1, proportional_n))
    total_allocated = sum(allocations.values())
    while total_allocated > n:
        candidates = [name for name, value in allocations.items() if value > 1]
        if not candidates:
            break
        largest = max(candidates, key=lambda name: allocations[name])
        allocations[largest] -= 1
        total_allocated -= 1
    while total_allocated < n:
        candidates = []
        for name, group in groups:
            if allocations[name] < len(group):
                candidates.append(name)
        if not candidates:
            break
        largest_remaining = max(candidates, key=lambda name: len(groups.get_group(name)) - allocations[name])
        allocations[largest_remaining] += 1
        total_allocated += 1
    pieces = []
    for i, (name, group) in enumerate(groups):
        take_n = allocations[name]
        pieces.append(group.sample(n=take_n, random_state=seed + i))
    result = pd.concat(pieces, ignore_index=True)
    return result.sample(frac=1.0, random_state=seed).reset_index(drop=True)
sample = stratified_sample(test_sampling, n=min(N_EXPLAIN, len(test_sampling)), stratum_col='_stratum', seed=RANDOM_STATE)
print(f'\nSHAP explanation sample: {len(sample):,}')
sample[[ID_COL, TARGET, LANDFORM_COL, VEGETATION_COL] + ALL_PREDICTORS].to_csv(OUT_DIR / 'ensemble_SHAP_sample.csv', index=False)
# Background data
background = train_df.sample(n=min(N_BACKGROUND, len(train_df)), random_state=RANDOM_STATE).copy()
background_X = background[ALL_PREDICTORS].to_numpy(dtype=float)
print('\nCalculating direct ensemble-level SHAP...')
print('Observed context is held fixed for each explained observation.')
print('This can take considerable time.')
n_features = len(ALL_PREDICTORS)
max_evals = N_PERMUTATION_CYCLES * (2 * n_features + 1)
print(f'Permutation max_evals per explanation: {max_evals}')
# SHAP by landform and vegetation context
sample['_context_key'] = sample[LANDFORM_COL].fillna('Missing').astype(str) + '|||' + sample[VEGETATION_COL].fillna('Missing').astype(str)
shap_matrix = np.full((len(sample), n_features), np.nan, dtype=float)
base_values = np.full(len(sample), np.nan, dtype=float)
model_outputs = np.full(len(sample), np.nan, dtype=float)
context_groups = sample.groupby('_context_key', sort=False).indices
for group_number, (context_key, positions) in enumerate(context_groups.items(), start=1):
    positions = np.asarray(positions, dtype=int)
    group_df = sample.iloc[positions]
    context_row = make_context_matrix(group_df.iloc[[0]])
    print(f'\nContext group {group_number}/{len(context_groups)}')
    print(f'  {context_key}')
    print(f'  n = {len(group_df)}')

    def context_fixed_predict(X_perturbed, fixed_context=context_row):
        X_perturbed = np.asarray(X_perturbed, dtype=float)
        repeated_context = np.repeat(fixed_context, repeats=X_perturbed.shape[0], axis=0)
        return ensemble_predict_with_context(X_perturbed, repeated_context)
    masker = shap.maskers.Independent(background_X, max_samples=len(background_X))
    explainer = shap.PermutationExplainer(context_fixed_predict, masker, feature_names=ALL_PREDICTORS, seed=RANDOM_STATE)
    X_group = group_df[ALL_PREDICTORS].to_numpy(dtype=float)
    explanation = explainer(X_group, max_evals=max_evals, batch_size='auto', silent=False)
    values = np.asarray(explanation.values)
    if values.ndim != 2 or values.shape != (len(group_df), n_features):
        raise RuntimeError(f'Unexpected SHAP output shape {values.shape}; expected ({len(group_df)}, {n_features}).')
    shap_matrix[positions, :] = values
    bv = np.asarray(explanation.base_values).reshape(-1)
    base_values[positions] = bv
    model_outputs[positions] = context_fixed_predict(X_group)
if np.isnan(shap_matrix).any():
    raise RuntimeError('SHAP matrix contains missing values. At least one observation was not explained.')
if np.isnan(base_values).any():
    raise RuntimeError('SHAP base values contain missing values.')
np.save(OUT_DIR / 'ensemble_SHAP_values.npy', shap_matrix)
pd.DataFrame({ID_COL: sample[ID_COL].values, 'base_value': base_values, 'ensemble_probability': model_outputs}).to_csv(OUT_DIR / 'ensemble_SHAP_expected_values.csv', index=False)
# SHAP additivity
reconstructed_probability = base_values + shap_matrix.sum(axis=1)
additivity_error = reconstructed_probability - model_outputs
additivity = pd.DataFrame({ID_COL: sample[ID_COL].values, 'base_value': base_values, 'sum_shap': shap_matrix.sum(axis=1), 'reconstructed_output': reconstructed_probability, 'actual_ensemble_probability': model_outputs, 'absolute_error': np.abs(additivity_error)})
additivity.to_csv(OUT_DIR / 'ensemble_SHAP_additivity_check.csv', index=False)
mean_additivity_error = additivity['absolute_error'].mean()
max_additivity_error = additivity['absolute_error'].max()
print('\nSHAP reconstruction check:')
print(f'Mean absolute error: {mean_additivity_error:.8f}')
print(f'Maximum absolute error: {max_additivity_error:.8f}')
# Feature and domain importance
mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
global_importance = pd.DataFrame({'feature': ALL_PREDICTORS, 'mean_abs_shap': mean_abs_shap})
global_importance['domain'] = global_importance['feature'].map(DOMAIN_MAP)
global_importance = global_importance.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
global_total = global_importance['mean_abs_shap'].sum()
global_importance['normalised_abs_shap'] = global_importance['mean_abs_shap'] / global_total
global_importance.to_csv(OUT_DIR / 'ensemble_SHAP_global_importance.csv', index=False)
domain_attribution = global_importance.groupby('domain', as_index=False)['mean_abs_shap'].sum()
domain_total = domain_attribution['mean_abs_shap'].sum()
domain_attribution['normalised_abs_shap'] = domain_attribution['mean_abs_shap'] / domain_total
domain_attribution = domain_attribution.sort_values('normalised_abs_shap', ascending=False)
domain_attribution.to_csv(OUT_DIR / 'ensemble_SHAP_domain_attribution.csv', index=False)
# Context-specific importance
context_feature_rows = []
context_domain_rows = []
for context_type, column in [('landform', LANDFORM_COL), ('vegetation', VEGETATION_COL)]:
    context_values = sample[column].fillna('Missing').astype(str)
    for level in sorted(context_values.unique()):
        mask = (context_values == level).to_numpy()
        n_context = int(mask.sum())
        if n_context < MIN_CONTEXT_N:
            continue
        context_abs = np.abs(shap_matrix[mask]).mean(axis=0)
        context_total = context_abs.sum()
        if context_total <= 0:
            continue
        for feature, value in zip(ALL_PREDICTORS, context_abs):
            context_feature_rows.append({'context_type': context_type, 'context_level': level, 'n': n_context, 'feature': feature, 'domain': DOMAIN_MAP[feature], 'mean_abs_shap': float(value), 'normalised_abs_shap': float(value / context_total)})
        temporary = pd.DataFrame({'feature': ALL_PREDICTORS, 'mean_abs_shap': context_abs})
        temporary['domain'] = temporary['feature'].map(DOMAIN_MAP)
        domain_context = temporary.groupby('domain', as_index=False)['mean_abs_shap'].sum()
        domain_context_total = domain_context['mean_abs_shap'].sum()
        for _, row in domain_context.iterrows():
            context_domain_rows.append({'context_type': context_type, 'context_level': level, 'n': n_context, 'domain': row['domain'], 'mean_abs_shap': float(row['mean_abs_shap']), 'normalised_abs_shap': float(row['mean_abs_shap'] / domain_context_total)})
context_feature_df = pd.DataFrame(context_feature_rows)
context_feature_df.to_csv(OUT_DIR / 'ensemble_SHAP_context_importance.csv', index=False)
context_domain_df = pd.DataFrame(context_domain_rows)
context_domain_df.to_csv(OUT_DIR / 'ensemble_SHAP_context_domain_attribution.csv', index=False)
# Figures
top = global_importance.head(TOP_N).copy()
plt.figure(figsize=(8, 7))
plt.barh(top['feature'][::-1], top['mean_abs_shap'][::-1])
plt.xlabel('Mean |SHAP value|')
plt.ylabel('Environmental predictor')
plt.title('Environmental predictor attribution\nin the full-context ensemble')
plt.tight_layout()
plt.savefig(OUT_DIR / 'Fig_ensemble_SHAP_top20.png', dpi=400, bbox_inches='tight')
plt.close()
X_sample_df = sample[ALL_PREDICTORS].reset_index(drop=True)
ensemble_explanation = shap.Explanation(values=shap_matrix, base_values=base_values, data=X_sample_df.to_numpy(), feature_names=ALL_PREDICTORS)
plt.figure()
shap.plots.beeswarm(ensemble_explanation, max_display=TOP_N, show=False)
plt.title('Environmental predictor attribution\nin the full-context ensemble')
plt.tight_layout()
plt.savefig(OUT_DIR / 'Fig_ensemble_SHAP_beeswarm.png', dpi=400, bbox_inches='tight')
plt.close()
domain_plot = domain_attribution.sort_values('normalised_abs_shap', ascending=True)
plt.figure(figsize=(6, 4))
plt.barh(domain_plot['domain'], domain_plot['normalised_abs_shap'])
plt.xlabel('Proportion of total mean |SHAP|')
plt.ylabel('Environmental domain')
plt.title('Environmental-domain attribution\nin the full-context ensemble')
plt.tight_layout()
plt.savefig(OUT_DIR / 'Fig_ensemble_SHAP_domain_attribution.png', dpi=400, bbox_inches='tight')
plt.close()
print('\n' + '=' * 78)
print('GLOBAL ENSEMBLE SHAP RESULTS')
print('=' * 78)
print('\nTop 20 environmental predictors:')
print(global_importance[['feature', 'domain', 'mean_abs_shap', 'normalised_abs_shap']].head(TOP_N).round(6).to_string(index=False))
print('\nEnvironmental-domain attribution:')
print(domain_attribution.round(6).to_string(index=False))
top_predictors_text = global_importance[['feature', 'domain', 'mean_abs_shap', 'normalised_abs_shap']].head(TOP_N).round(6).to_string(index=False)
domain_text = domain_attribution.round(6).to_string(index=False)
summary = f"\nAppendix A direct ensemble-level SHAP\n=====================================\n\nModel explained\n---------------\nFinal full-context (FC) attention ensemble.\n\nPrediction pathway\n------------------\n31 environmental predictors\n-> final fitted RF/XGB/SVM/MLP pipelines\n-> 8 two-class probability inputs\n-> observed landform + vegetation context\n-> context-conditioned attention\n-> weighted two-class probability vectors\n-> final Dense(2, softmax)\n-> P(high severity)\n\nImportant methodological distinction\n------------------------------------\nThese SHAP values explain the COMPLETE fitted ensemble prediction function.\n\nThey are NOT attention-weighted averages of SHAP values from the four\nindividual base classifiers.\n\nContext treatment\n-----------------\nFor each explained observation, its observed landform and vegetation\ncontext was held fixed while environmental predictors were perturbed.\n\nTherefore these SHAP values represent environmental-predictor attribution\nconditional on the observed landscape context.\n\nSampling\n--------\nHeld-out TEST observations available: {len(test_df):,}\nObservations explained: {len(sample):,}\nBackground observations from TRAIN: {len(background):,}\nEnvironmental predictors: {len(ALL_PREDICTORS)}\nPermutation cycles: {N_PERMUTATION_CYCLES}\nmax_evals: {max_evals}\n\nPipeline verification\n---------------------\nReconstructed FC test AUC: {test_auc:.6f}\n\nSHAP additivity / reconstruction\n--------------------------------\nMean absolute reconstruction error: {mean_additivity_error:.10f}\nMaximum absolute reconstruction error: {max_additivity_error:.10f}\n\nTop environmental predictors\n----------------------------\n{top_predictors_text}\n\nEnvironmental-domain attribution\n--------------------------------\n{domain_text}\n\nInterpretation\n--------------\nMean absolute SHAP values quantify attribution within the fitted FC ensemble.\nThey should not be interpreted as causal environmental effects.\n\nEnvironmental-domain SHAP proportions describe the distribution of ensemble\nfeature attribution and should not replace the separate environmental-domain\nablation experiment, which evaluates predictive information retained by\ndifferent predictor-domain combinations.\n\nAttention weights should be interpreted separately as the ensemble's\nallocation of classifier contribution across landscape contexts.\n\nOutputs\n-------\n{OUT_DIR}\n"
(OUT_DIR / 'RUN_SUMMARY.txt').write_text(summary, encoding='utf-8')
print('\n' + summary)
print('\nEnsemble-level SHAP completed successfully.')
