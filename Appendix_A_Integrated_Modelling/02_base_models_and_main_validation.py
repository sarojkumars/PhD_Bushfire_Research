# -*- coding: utf-8 -*-
"""
Appendix A validation pipeline
=============================

Purpose
-------
Leakage-controlled validation for Appendix A fire-severity modelling.

Core workflow
-------------
1. Load the full modelling dataset once.
2. Create ONE deterministic, stratified 80/20 train/test split and save row IDs.
3. Fit preprocessing on TRAIN only.
4. Tune four base classifiers using TRAIN only.
5. Generate out-of-fold (OOF) probabilities for meta-model training.
6. Refit tuned base classifiers on all TRAIN data.
7. Generate final base-model probabilities for the untouched TEST set.
8. Compare four attention ensembles using identical OOF base probabilities:
      CF = context free
      TC = topographic context
      VC = vegetation context
      FC = full topographic + vegetation context
9. Evaluate all final models once on the untouched TEST set.
10. Report context-specific performance without fitting separate scenario models.
11. Extract attention weights by landform and vegetation context.
12. Run environmental-domain ablation using XGBoost:
      M, T, FV, M+T, M+FV, T+FV, M+T+FV
13. Save all metrics, predictions, split IDs, hyperparameters and metadata.

"""

from __future__ import annotations

import os
import json
import random
import warnings
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_predict,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
)
from xgboost import XGBClassifier

# TensorFlow/Keras is used only for the attention meta-model.
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Multiply, Lambda
from tensorflow.keras.callbacks import EarlyStopping


# ---------------------------------------------------------------------
# 0. USER SETTINGS
# ---------------------------------------------------------------------

# Centralised paths: data are one folder above this code directory.
SCRIPT_DIR = Path(__file__).resolve().parent
from appendix_config import DATA_FILE, VAL_DIR
OUTPUT_DIR = VAL_DIR

ID_COL = "pt_id"
TARGET = "SNIC2clsN"
LANDFORM_COL = "LF4Cls"
VEGETATION_COL = "Veg5Cls"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_FOLDS = 5
N_JOBS = -1

# Four intended final base models.
BASE_MODEL_NAMES = ["RF", "XGB", "SVM", "MLP"]

# Attention model: fixed architecture across CF/TC/VC/FC.
# We intentionally DO NOT tune architecture on the held-out test set.
ATTENTION_HIDDEN_1 = 64
ATTENTION_HIDDEN_2 = 32
ATTENTION_EPOCHS = 100
ATTENTION_BATCH_SIZE = 128
ATTENTION_VALIDATION_SPLIT = 0.15
ATTENTION_PATIENCE = 10

# Set True for a quick pipeline test. Never use QUICK_TEST results in thesis.
QUICK_TEST = False
QUICK_TEST_N = 12000

# Reuse saved tuned models on reruns if desired.
REUSE_TUNED_BASE_MODELS = False

# ---------------------------------------------------------------------
# Predictor definitions based on the supplied Appendix A v5 scripts/data.
# ---------------------------------------------------------------------

METEOROLOGY = [
    "avgMaxTempSn",
    "avgSurfPresSn",
    "TotRainSn",
    "TotRainYr",
    "avgRHtmaxSn",
    "avgV10Sn",
    "avgU10Sn",
    "avgAbsWndSn",
    "avgSwvL1Sn",
    "SoilMois_0to10",
]

TOPOGRAPHY = [
    "DEM",
    "SlpPerc",
    "AspCosine",
    "TWI",
    "SolDur",
    "SolDifRad",
    "ConvIndx",
    "ProfCurv",
    "PlanCurv",
    "LocCurv",
    "DwnSlpCurv",
    "UpSlpCurv",
]

FUEL_VEGETATION = [
    "NDMI_mean",
    "LAI",
    "LFMC",
    "EVI",
    "DFFM_silo",
    "FCOV_BS",
    "FCOV_PV",
    "Flamm",
    "CHM",
]

ALL_PREDICTORS = METEOROLOGY + TOPOGRAPHY + FUEL_VEGETATION

DOMAIN_SETS = {
    "M": METEOROLOGY,
    "T": TOPOGRAPHY,
    "FV": FUEL_VEGETATION,
    "M+T": METEOROLOGY + TOPOGRAPHY,
    "M+FV": METEOROLOGY + FUEL_VEGETATION,
    "T+FV": TOPOGRAPHY + FUEL_VEGETATION,
    "M+T+FV": ALL_PREDICTORS,
}


# ---------------------------------------------------------------------
# 1. REPRODUCIBILITY
# ---------------------------------------------------------------------

def set_all_seeds(seed: int = RANDOM_STATE) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def ensure_columns(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}")


# ---------------------------------------------------------------------
# 2. METRICS
# ---------------------------------------------------------------------

def binary_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob).reshape(-1)
    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    return {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, pred),
        "precision_class1": precision_score(y_true, pred, zero_division=0),
        "recall_class1": recall_score(y_true, pred, zero_division=0),
        "f1_class1": f1_score(y_true, pred, zero_division=0),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, pred, average="weighted", zero_division=0),
        "auc": roc_auc_score(y_true, prob) if len(np.unique(y_true)) == 2 else np.nan,
        "mcc": matthews_corrcoef(y_true, pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ---------------------------------------------------------------------
# 3. DATA LOADING AND ONE FIXED SPLIT
# ---------------------------------------------------------------------

def load_and_split():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find:\n{DATA_FILE}\n\n"
            "Edit DATA_FILE near the top of this script."
        )

    df = pd.read_csv(DATA_FILE)

    ensure_columns(
        df,
        [ID_COL, TARGET, LANDFORM_COL, VEGETATION_COL] + ALL_PREDICTORS,
        "required",
    )

    # Keep rows with target and ID. Predictors are imputed within training pipelines.
    df = df.dropna(subset=[ID_COL, TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET} must be binary 0/1. Found: {sorted(df[TARGET].unique())}")

    # Avoid duplicate point IDs compromising traceability.
    if df[ID_COL].duplicated().any():
        ndup = int(df[ID_COL].duplicated().sum())
        raise ValueError(f"{ID_COL} contains {ndup} duplicate values. Resolve before validation.")

    # Optional development-only subsample, deterministic and stratified.
    if QUICK_TEST and len(df) > QUICK_TEST_N:
        dev, _ = train_test_split(
            df,
            train_size=QUICK_TEST_N,
            stratify=df[TARGET],
            random_state=RANDOM_STATE,
        )
        df = dev.copy()

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET],
        random_state=RANDOM_STATE,
    )

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Save the split immediately. These IDs define the validation experiment.
    split_table = pd.concat(
        [
            train_df[[ID_COL, TARGET]].assign(split="TRAIN"),
            test_df[[ID_COL, TARGET]].assign(split="TEST"),
        ],
        ignore_index=True,
    )
    split_table.to_csv(OUTPUT_DIR / "01_fixed_split_ids.csv", index=False)

    return df, train_df, test_df


# ---------------------------------------------------------------------
# 4. BASE-MODEL PIPELINES AND TUNING
# ---------------------------------------------------------------------

def make_pipeline(estimator):
    # Imputation and scaling are fitted inside each CV training fold.
    # Scaling tree models is unnecessary but harmless and keeps a common,
    # leakage-safe pipeline for all four algorithms.
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ]
    )


def model_searches():
    """
    Moderate grids: enough to reproduce a defensible comparison without turning
    Appendix A into another large algorithm-search exercise.
    """
    searches = {}

    rf = make_pipeline(
        RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            class_weight=None,
        )
    )
    searches["RF"] = GridSearchCV(
        rf,
        param_grid={
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 20, 30],
            "model__min_samples_split": [2, 5],
        },
        scoring="roc_auc",
        cv=StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=N_JOBS,
        refit=True,
        verbose=1,
    )

    xgb = make_pipeline(
        XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=max(1, os.cpu_count() or 1),
        )
    )
    searches["XGB"] = GridSearchCV(
        xgb,
        param_grid={
            "model__n_estimators": [200, 400],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [3, 6],
            "model__subsample": [0.8, 1.0],
        },
        scoring="roc_auc",
        cv=StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=N_JOBS,
        refit=True,
        verbose=1,
    )

    svm = make_pipeline(
        SVC(
            probability=True,
            random_state=RANDOM_STATE,
            cache_size=4000,
        )
    )
    searches["SVM"] = GridSearchCV(
        svm,
        param_grid={
            "model__C": [0.5, 1, 5, 10],
            "model__gamma": ["scale", "auto"],
            "model__kernel": ["rbf"],
        },
        scoring="roc_auc",
        cv=StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=N_JOBS,
        refit=True,
        verbose=1,
    )

    mlp = make_pipeline(
        MLPClassifier(
            random_state=RANDOM_STATE,
            max_iter=1500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )
    )
    searches["MLP"] = RandomizedSearchCV(
        mlp,
        param_distributions={
            "model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64)],
            "model__activation": ["relu", "tanh"],
            "model__alpha": [0.0001, 0.001, 0.01],
            "model__learning_rate_init": [0.0005, 0.001, 0.005],
        },
        n_iter=12,
        scoring="roc_auc",
        cv=StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=N_JOBS,
        refit=True,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    return searches


def tune_base_models(X_train: pd.DataFrame, y_train: pd.Series):
    tuned = {}
    tuning_rows = []

    searches = model_searches()

    for name in BASE_MODEL_NAMES:
        model_path = OUTPUT_DIR / f"base_{name}_tuned.joblib"

        if REUSE_TUNED_BASE_MODELS and model_path.exists():
            estimator = joblib.load(model_path)
            tuned[name] = estimator
            tuning_rows.append(
                {"model": name, "best_cv_auc": np.nan, "best_params": "REUSED_SAVED_MODEL"}
            )
            continue

        print(f"\n{'='*70}\nTUNING BASE MODEL: {name}\n{'='*70}")
        search = searches[name]
        search.fit(X_train, y_train)

        tuned[name] = search.best_estimator_
        joblib.dump(search.best_estimator_, model_path)

        tuning_rows.append(
            {
                "model": name,
                "best_cv_auc": float(search.best_score_),
                "best_params": json.dumps(search.best_params_, default=str),
            }
        )

    pd.DataFrame(tuning_rows).to_csv(
        OUTPUT_DIR / "02_base_model_tuning.csv", index=False
    )
    return tuned


# ---------------------------------------------------------------------
# 5. OOF PROBABILITIES + FINAL TEST PROBABILITIES
# ---------------------------------------------------------------------

def generate_oof_and_test_probs(tuned_models, X_train, y_train, X_test):
    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof = pd.DataFrame(index=X_train.index)
    test = pd.DataFrame(index=X_test.index)

    for name, estimator in tuned_models.items():
        print(f"\nGenerating leakage-free OOF probabilities: {name}")

        # Each training observation is predicted by a model that did not fit it.
        oof_prob = cross_val_predict(
            clone(estimator),
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=N_JOBS,
        )[:, 1]

        # Final base estimator fitted on all TRAIN observations only.
        final_estimator = clone(estimator)
        final_estimator.fit(X_train, y_train)
        joblib.dump(final_estimator, OUTPUT_DIR / f"base_{name}_final.joblib")

        test_prob = final_estimator.predict_proba(X_test)[:, 1]

        oof[name] = oof_prob
        test[name] = test_prob

    oof.to_csv(OUTPUT_DIR / "03_oof_base_probabilities.csv", index=False)
    test.to_csv(OUTPUT_DIR / "04_test_base_probabilities.csv", index=False)

    return oof, test


# ---------------------------------------------------------------------
# 6. CONTEXT MATRICES
# ---------------------------------------------------------------------

def one_hot_context(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Create context encodings using TRAIN categories, then align TEST columns.
    We keep topography and vegetation separate for TC/VC ablations.
    """
    topo_train = pd.get_dummies(
        train_df[LANDFORM_COL].astype(str),
        prefix="topo",
        dtype=float,
    )
    topo_test = pd.get_dummies(
        test_df[LANDFORM_COL].astype(str),
        prefix="topo",
        dtype=float,
    ).reindex(columns=topo_train.columns, fill_value=0.0)

    veg_train = pd.get_dummies(
        train_df[VEGETATION_COL].astype(str),
        prefix="veg",
        dtype=float,
    )
    veg_test = pd.get_dummies(
        test_df[VEGETATION_COL].astype(str),
        prefix="veg",
        dtype=float,
    ).reindex(columns=veg_train.columns, fill_value=0.0)

    return (
        topo_train.to_numpy(np.float32),
        topo_test.to_numpy(np.float32),
        list(topo_train.columns),
        veg_train.to_numpy(np.float32),
        veg_test.to_numpy(np.float32),
        list(veg_train.columns),
    )


# ---------------------------------------------------------------------
# 7. ATTENTION META-MODEL
# ---------------------------------------------------------------------

def build_attention_model(n_models: int, n_context: int, seed: int):
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    prob_input = Input(shape=(n_models,), name="base_probabilities")

    if n_context > 0:
        context_input = Input(shape=(n_context,), name="context")
        z = Concatenate(name="prob_context_concat")([prob_input, context_input])
        model_inputs = [prob_input, context_input]
    else:
        z = prob_input
        model_inputs = prob_input

    z = Dense(ATTENTION_HIDDEN_1, activation="relu", name="dense_1")(z)
    z = Dense(ATTENTION_HIDDEN_2, activation="relu", name="dense_2")(z)

    weights = Dense(
        n_models,
        activation="softmax",
        name="attention_weights",
    )(z)

    weighted_probs = Multiply(name="weighted_probabilities")([prob_input, weights])

    # Convex weighted sum of class-1 probabilities.
    output = Lambda(
        lambda x: tf.reduce_sum(x, axis=1, keepdims=True),
        name="ensemble_probability",
    )(weighted_probs)

    model = Model(inputs=model_inputs, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )
    return model


def fit_attention_variant(
    variant_name,
    oof_probs,
    y_train,
    test_probs,
    context_train=None,
    context_test=None,
    seed_offset=0,
):
    """
    The held-out TEST set is not passed to fit().
    Keras takes an internal validation fraction from OOF meta-training data only.
    """
    seed = RANDOM_STATE + seed_offset
    n_context = 0 if context_train is None else context_train.shape[1]

    model = build_attention_model(
        n_models=oof_probs.shape[1],
        n_context=n_context,
        seed=seed,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=ATTENTION_PATIENCE,
            restore_best_weights=True,
            min_delta=1e-4,
        )
    ]

    fit_kwargs = dict(
        x=oof_probs.astype(np.float32),
        y=np.asarray(y_train).astype(np.float32),
        epochs=ATTENTION_EPOCHS,
        batch_size=ATTENTION_BATCH_SIZE,
        validation_split=ATTENTION_VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=0,
        shuffle=True,
    )

    if context_train is not None:
        fit_kwargs["x"] = [
            oof_probs.astype(np.float32),
            context_train.astype(np.float32),
        ]

    history = model.fit(**fit_kwargs)

    if context_test is None:
        test_input = test_probs.astype(np.float32)
    else:
        test_input = [
            test_probs.astype(np.float32),
            context_test.astype(np.float32),
        ]

    test_prob = model.predict(test_input, verbose=0).reshape(-1)

    model.save(OUTPUT_DIR / f"attention_{variant_name}.keras")

    pd.DataFrame(history.history).to_csv(
        OUTPUT_DIR / f"attention_{variant_name}_training_history.csv",
        index=False,
    )

    # Attention weights for TEST observations.
    weight_model = Model(
        inputs=model.inputs,
        outputs=model.get_layer("attention_weights").output,
    )
    weights = weight_model.predict(test_input, verbose=0)

    return model, test_prob, weights


# ---------------------------------------------------------------------
# 8. CONTEXT ABLATION: CF / TC / VC / FC
# ---------------------------------------------------------------------

def run_context_ablation(
    train_df,
    test_df,
    y_train,
    y_test,
    oof_df,
    test_prob_df,
):
    (
        topo_train,
        topo_test,
        topo_names,
        veg_train,
        veg_test,
        veg_names,
    ) = one_hot_context(train_df, test_df)

    full_train = np.hstack([topo_train, veg_train])
    full_test = np.hstack([topo_test, veg_test])

    variants = {
        "CF": (None, None),
        "TC": (topo_train, topo_test),
        "VC": (veg_train, veg_test),
        "FC": (full_train, full_test),
    }

    rows = []
    predictions = test_df[[ID_COL, TARGET, LANDFORM_COL, VEGETATION_COL]].copy()
    attention_tables = {}

    for i, (variant, (ctx_train, ctx_test)) in enumerate(variants.items()):
        print(f"\nTraining attention variant: {variant}")

        _, prob, weights = fit_attention_variant(
            variant_name=variant,
            oof_probs=oof_df.to_numpy(),
            y_train=y_train,
            test_probs=test_prob_df.to_numpy(),
            context_train=ctx_train,
            context_test=ctx_test,
            seed_offset=i,
        )

        m = binary_metrics(y_test, prob)
        m.update({"analysis": "context_ablation", "model": variant})
        rows.append(m)

        predictions[f"prob_{variant}"] = prob
        predictions[f"pred_{variant}"] = (prob >= 0.5).astype(int)

        w = pd.DataFrame(weights, columns=BASE_MODEL_NAMES)
        w.insert(0, VEGETATION_COL, test_df[VEGETATION_COL].values)
        w.insert(0, LANDFORM_COL, test_df[LANDFORM_COL].values)
        w.insert(0, TARGET, y_test.values)
        w.insert(0, ID_COL, test_df[ID_COL].values)
        attention_tables[variant] = w
        w.to_csv(OUTPUT_DIR / f"attention_weights_{variant}.csv", index=False)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUTPUT_DIR / "05_context_ablation_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "06_context_ablation_predictions.csv", index=False)

    # Context-specific evaluation: NO retraining; subset the same held-out TEST predictions.
    context_rows = []
    for variant in variants:
        prob = predictions[f"prob_{variant}"].to_numpy()

        for context_type, col in [
            ("landform", LANDFORM_COL),
            ("vegetation", VEGETATION_COL),
        ]:
            for level in sorted(test_df[col].dropna().astype(str).unique()):
                mask = test_df[col].astype(str).to_numpy() == level
                if mask.sum() < 2:
                    continue
                yy = y_test.to_numpy()[mask]
                pp = prob[mask]
                mm = binary_metrics(yy, pp)
                mm.update(
                    {
                        "variant": variant,
                        "context_type": context_type,
                        "context_level": level,
                    }
                )
                context_rows.append(mm)

    pd.DataFrame(context_rows).to_csv(
        OUTPUT_DIR / "07_context_specific_test_metrics.csv",
        index=False,
    )

    # Summarise attention weights for the full-context model.
    fc_weights = attention_tables["FC"]

    long_summary = []
    for context_type, col in [
        ("landform", LANDFORM_COL),
        ("vegetation", VEGETATION_COL),
    ]:
        for level, g in fc_weights.groupby(col, dropna=False):
            for model_name in BASE_MODEL_NAMES:
                long_summary.append(
                    {
                        "context_type": context_type,
                        "context_level": level,
                        "base_model": model_name,
                        "mean_attention_weight": g[model_name].mean(),
                        "sd_attention_weight": g[model_name].std(),
                        "n": len(g),
                    }
                )

    pd.DataFrame(long_summary).to_csv(
        OUTPUT_DIR / "08_FC_attention_weights_by_context.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "topographic_context_columns": pd.Series(topo_names),
            "vegetation_context_columns": pd.Series(veg_names),
        }
    ).to_csv(OUTPUT_DIR / "context_column_names.csv", index=False)

    return metrics_df, predictions


# ---------------------------------------------------------------------
# 9. BASE-MODEL FINAL TEST METRICS
# ---------------------------------------------------------------------

def evaluate_base_models(y_test, test_prob_df):
    rows = []
    for name in BASE_MODEL_NAMES:
        m = binary_metrics(y_test, test_prob_df[name].to_numpy())
        m.update({"analysis": "base_model", "model": name})
        rows.append(m)

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_DIR / "09_base_model_test_metrics.csv", index=False)
    return out


# ---------------------------------------------------------------------
# 10. ENVIRONMENTAL-DOMAIN ABLATION
# ---------------------------------------------------------------------

def run_domain_ablation(train_df, test_df, y_train, y_test):
    """
    One fixed nonlinear learner (XGB) is used here so the experiment tests
    environmental information, not classifier architecture.

    Hyperparameters are selected independently within TRAIN for each domain set.
    TEST remains untouched until the selected model is evaluated.
    """
    rows = []
    predictions = test_df[[ID_COL, TARGET]].copy()

    cv = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for label, cols in DOMAIN_SETS.items():
        print(f"\nDomain ablation: {label} ({len(cols)} predictors)")

        ensure_columns(train_df, cols, f"{label} domain")

        pipe = make_pipeline(
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=max(1, os.cpu_count() or 1),
            )
        )

        search = GridSearchCV(
            pipe,
            param_grid={
                "model__n_estimators": [200, 400],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [3, 6],
                "model__subsample": [0.8, 1.0],
            },
            scoring="roc_auc",
            cv=cv,
            n_jobs=N_JOBS,
            refit=True,
            verbose=1,
        )

        search.fit(train_df[cols], y_train)
        prob = search.best_estimator_.predict_proba(test_df[cols])[:, 1]

        m = binary_metrics(y_test, prob)
        m.update(
            {
                "analysis": "domain_ablation",
                "domain_set": label,
                "n_predictors": len(cols),
                "best_cv_auc": float(search.best_score_),
                "best_params": json.dumps(search.best_params_, default=str),
            }
        )
        rows.append(m)

        predictions[f"prob_{label.replace('+', '_')}"] = prob

        safe_label = label.replace("+", "_")
        joblib.dump(
            search.best_estimator_,
            OUTPUT_DIR / f"domain_ablation_XGB_{safe_label}.joblib",
        )

    pd.DataFrame(rows).to_csv(
        OUTPUT_DIR / "10_environmental_domain_ablation_metrics.csv",
        index=False,
    )
    predictions.to_csv(
        OUTPUT_DIR / "11_environmental_domain_ablation_predictions.csv",
        index=False,
    )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 11. ANALYSIS METADATA
# ---------------------------------------------------------------------

def save_metadata(df, train_df, test_df):
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "data_file": str(DATA_FILE),
        "n_total_eligible": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "test_fraction": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_cv_folds": N_FOLDS,
        "target": TARGET,
        "id_column": ID_COL,
        "landform_column": LANDFORM_COL,
        "vegetation_column": VEGETATION_COL,
        "n_predictors": len(ALL_PREDICTORS),
        "n_meteorology": len(METEOROLOGY),
        "n_topography": len(TOPOGRAPHY),
        "n_fuel_vegetation": len(FUEL_VEGETATION),
        "base_models": BASE_MODEL_NAMES,
        "quick_test": QUICK_TEST,
        "validation_statement": (
            "Base models tuned using training data only; OOF predictions generated "
            "within training data for meta-model fitting; held-out test data used "
            "only for final evaluation."
        ),
    }

    with open(OUTPUT_DIR / "00_validation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------
# 12. MAIN
# ---------------------------------------------------------------------

def main():
    warnings.filterwarnings("ignore")
    set_all_seeds()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("APPENDIX A LEAKAGE-CONTROLLED VALIDATION")
    print("=" * 78)

    df, train_df, test_df = load_and_split()
    save_metadata(df, train_df, test_df)

    print(f"\nTotal observations: {len(df):,}")
    print(f"TRAIN: {len(train_df):,}")
    print(f"TEST:  {len(test_df):,}")
    print("\nClass distribution:")
    print(pd.crosstab(
        pd.Series(["TRAIN"] * len(train_df) + ["TEST"] * len(test_df), name="split"),
        pd.concat([train_df[TARGET], test_df[TARGET]], ignore_index=True),
        normalize="index",
    ))

    X_train = train_df[ALL_PREDICTORS].copy()
    X_test = test_df[ALL_PREDICTORS].copy()
    y_train = train_df[TARGET].astype(int)
    y_test = test_df[TARGET].astype(int)

    # A. Base models
    tuned_models = tune_base_models(X_train, y_train)

    # B. Leakage-free OOF meta-features + untouched final test probabilities
    oof_df, test_prob_df = generate_oof_and_test_probs(
        tuned_models,
        X_train,
        y_train,
        X_test,
    )

    base_metrics = evaluate_base_models(y_test, test_prob_df)

    # C. Central context/no-context ablation
    context_metrics, context_predictions = run_context_ablation(
        train_df=train_df,
        test_df=test_df,
        y_train=y_train,
        y_test=y_test,
        oof_df=oof_df,
        test_prob_df=test_prob_df,
    )

    # D. Environmental-domain ablation
    domain_metrics = run_domain_ablation(
        train_df=train_df,
        test_df=test_df,
        y_train=y_train,
        y_test=y_test,
    )

    # E. One compact summary table
    summary = pd.concat(
        [
            base_metrics,
            context_metrics,
        ],
        ignore_index=True,
        sort=False,
    )
    summary.to_csv(OUTPUT_DIR / "12_main_model_comparison.csv", index=False)

    print("\n" + "=" * 78)
    print("VALIDATION COMPLETE")
    print("=" * 78)
    print(f"\nOutputs saved to:\n{OUTPUT_DIR}")

    print("\nMain model comparison:")
    show_cols = [
        c for c in
        ["analysis", "model", "n", "auc", "f1_macro", "f1_weighted", "mcc", "accuracy"]
        if c in summary.columns
    ]
    print(summary[show_cols].round(4).to_string(index=False))

    print("\nEnvironmental-domain ablation:")
    show_domain = [
        c for c in
        ["domain_set", "n_predictors", "auc", "f1_macro", "mcc", "accuracy"]
        if c in domain_metrics.columns
    ]
    print(domain_metrics[show_domain].round(4).to_string(index=False))

    print(
        "\nIMPORTANT: Do not interpret these results as spatially independent "
        "validation. A spatial-block sensitivity analysis requires coordinates "
        "or a spatial grouping variable joined to pt_id."
    )


if __name__ == "__main__":
    main()
