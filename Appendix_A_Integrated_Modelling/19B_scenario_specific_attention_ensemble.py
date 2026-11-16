# -*- coding: utf-8 -*-
"""
19B_scenario_specific_attention_ensemble.py

Purpose
-------
Compare, for each Appendix A landform and vegetation scenario:

A) GLOBAL FC ensemble evaluated within that scenario
versus
B) SCENARIO-SPECIFIC ensemble trained only on training observations belonging
   to that scenario and evaluated only on the corresponding untouched test
   observations.

This is a supplementary validation of the existing Appendix A framework.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix
)
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from appendix_config import (
    DATA_FILE, VAL_DIR, ID_COL, TARGET, LANDFORM_COL, VEGETATION_COL,
    RANDOM_STATE, BASE_MODELS
)

# ---------------------------------------------------------------------
# 0. PATHS / SETTINGS
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT19_DIR = SCRIPT_DIR / "appendix_innovative_ensemble"
OUT_DIR = SCRIPT_DIR / "appendix_scenario_specific_ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OOF_FILE = VAL_DIR / "03_oof_base_probabilities.csv"
TEST_PROB_FILE = VAL_DIR / "04_test_base_probabilities.csv"
SPLIT_FILE = VAL_DIR / "01_fixed_split_ids.csv"

SELECTED_ARCH_FILE = SCRIPT19_DIR / "04_selected_architecture.json"
GLOBAL_PRED_FILE = SCRIPT19_DIR / "05_final_test_predictions.csv"

# The four landform and four ecological scenarios used in Appendix A.
LANDFORM_SCENARIOS = ["Peak", "UpSlp", "LwSlp", "Valy"]
VEGETATION_SCENARIOS = ["RFVT", "EFWL", "OFWL", "SHMW"]

# Internal early stopping is based ONLY on scenario training observations.
SCENARIO_VALID_SIZE = 0.20
MAX_EPOCHS = 200
PATIENCE = 15
MIN_DELTA = 1e-4

# Paired bootstrap for scenario-specific vs global-FC comparison.
N_BOOTSTRAP = 2000

# Small groups are retained, but explicitly flagged.
MIN_TEST_N_WARNING = 500
MIN_CLASS_N_FOR_BOOTSTRAP = 10


# ---------------------------------------------------------------------
# 1. REPRODUCIBILITY
# ---------------------------------------------------------------------

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


# ---------------------------------------------------------------------
# 2. METRICS
# ---------------------------------------------------------------------

def calculate_metrics(y_true, prob, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float).reshape(-1)
    pred = (prob >= threshold).astype(int)

    if len(np.unique(y_true)) < 2:
        auc = np.nan
    else:
        auc = roc_auc_score(y_true, prob)

    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    return {
        "n": int(len(y_true)),
        "n_class0": int(np.sum(y_true == 0)),
        "n_class1": int(np.sum(y_true == 1)),
        "prevalence_class1": float(np.mean(y_true)),
        "accuracy": accuracy_score(y_true, pred),
        "precision_class1": precision_score(y_true, pred, zero_division=0),
        "recall_class1": recall_score(y_true, pred, zero_division=0),
        "f1_class1": f1_score(y_true, pred, zero_division=0),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, pred, average="weighted", zero_division=0),
        "auc": auc,
        "mcc": matthews_corrcoef(y_true, pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# ---------------------------------------------------------------------
# 3. ORIGINAL ATTENTION ARCHITECTURE
# ---------------------------------------------------------------------

def build_original_attention(
    n_models,
    n_context,
    hidden_units=(128, 64),
    dropout=0.0,
    l2_value=0.0,
    learning_rate=3e-4,
    seed=42,
):
    """
    Same core architecture used by corrected Script 19:
    full 2-class base probabilities -> attention network -> dynamic model
    weights -> weighted probability vectors -> Dense(2, softmax).
    """
    tf.keras.backend.clear_session()
    set_seed(seed)

    n_prob_features = n_models * 2
    prob_input = Input(
        shape=(n_prob_features,),
        name="base_two_class_probabilities"
    )

    if n_context > 0:
        context_input = Input(shape=(n_context,), name="context")
        z = Concatenate(name="prob_context_concat")(
            [prob_input, context_input]
        )
        inputs = [prob_input, context_input]
    else:
        z = prob_input
        inputs = prob_input

    reg = regularizers.l2(l2_value) if l2_value > 0 else None

    for i, units in enumerate(hidden_units, start=1):
        z = Dense(
            units,
            activation="relu",
            kernel_regularizer=reg,
            name=f"attention_dense_{i}",
        )(z)
        if dropout > 0:
            z = Dropout(
                dropout,
                name=f"attention_dropout_{i}"
            )(z)

    attention = Dense(
        n_models,
        activation="softmax",
        kernel_regularizer=reg,
        name="attention_weights",
    )(z)

    prob_matrix = Lambda(
        lambda x: tf.reshape(x, (-1, n_models, 2)),
        name="reshape_model_probabilities",
    )(prob_input)

    attention_expanded = Lambda(
        lambda x: tf.expand_dims(x, axis=-1),
        name="expand_attention",
    )(attention)

    weighted_matrix = Lambda(
        lambda xs: xs[0] * xs[1],
        name="weighted_model_probabilities",
    )([prob_matrix, attention_expanded])

    combined = Lambda(
        lambda x: tf.reduce_sum(x, axis=1),
        name="combined_weighted_probabilities",
    )(weighted_matrix)

    output = Dense(
        2,
        activation="softmax",
        kernel_regularizer=reg,
        name="final_output",
    )(combined)

    model = Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def pack_inputs(prob8, context):
    if context is None:
        return prob8.astype(np.float32)
    return [
        prob8.astype(np.float32),
        context.astype(np.float32)
    ]


def parse_hidden(value):
    if isinstance(value, (list, tuple)):
        return tuple(int(x) for x in value)
    return tuple(int(x) for x in str(value).split("-"))


# ---------------------------------------------------------------------
# 4. LOAD AND ALIGN DATA
# ---------------------------------------------------------------------

def load_data():
    required = [
        DATA_FILE, OOF_FILE, TEST_PROB_FILE, SPLIT_FILE,
        SELECTED_ARCH_FILE, GLOBAL_PRED_FILE
    ]
    missing = [str(p) for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing)
        )

    data = pd.read_csv(DATA_FILE)
    split = pd.read_csv(SPLIT_FILE)
    oof = pd.read_csv(OOF_FILE)
    testp = pd.read_csv(TEST_PROB_FILE)
    global_pred = pd.read_csv(GLOBAL_PRED_FILE)

    with open(SELECTED_ARCH_FILE, "r", encoding="utf-8") as f:
        selected = json.load(f)

    # Check base probability columns.
    for model_name in BASE_MODELS:
        if model_name not in oof.columns:
            raise ValueError(
                f"{model_name} missing from {OOF_FILE}"
            )
        if model_name not in testp.columns:
            raise ValueError(
                f"{model_name} missing from {TEST_PROB_FILE}"
            )

    # Fixed split in exact saved order.
    train_ids = split.loc[
        split["split"] == "TRAIN", [ID_COL, TARGET]
    ].reset_index(drop=True)

    test_ids = split.loc[
        split["split"] == "TEST", [ID_COL, TARGET]
    ].reset_index(drop=True)

    if len(train_ids) != len(oof):
        raise ValueError(
            f"TRAIN/OOF mismatch: {len(train_ids)} vs {len(oof)}"
        )

    if len(test_ids) != len(testp):
        raise ValueError(
            f"TEST/probability mismatch: {len(test_ids)} vs {len(testp)}"
        )

    # Add context from master data.
    ctx = data[
        [ID_COL, LANDFORM_COL, VEGETATION_COL]
    ].drop_duplicates(subset=[ID_COL])

    train_info = train_ids.merge(
        ctx, on=ID_COL, how="left", validate="one_to_one"
    )
    test_info = test_ids.merge(
        ctx, on=ID_COL, how="left", validate="one_to_one"
    )

    y_train = train_info[TARGET].astype(int).to_numpy()
    y_test = test_info[TARGET].astype(int).to_numpy()

    # Reconstruct [P(class0), P(class1)] for each base model.
    def to_two_class(df):
        blocks = []
        for model_name in BASE_MODELS:
            p1 = np.clip(
                df[model_name].to_numpy(float),
                1e-7, 1 - 1e-7
            )
            blocks.append(
                np.column_stack([1.0 - p1, p1])
            )
        return np.hstack(blocks).astype(np.float32)

    train_prob8 = to_two_class(oof)
    test_prob8 = to_two_class(testp)

    # Align global FC predictions by pt_id whenever possible.
    if ID_COL in global_pred.columns:
        gp = global_pred[[ID_COL, "prob_FC"]].copy()
        test_global = test_info[[ID_COL]].merge(
            gp, on=ID_COL, how="left", validate="one_to_one"
        )
        if test_global["prob_FC"].isna().any():
            raise ValueError(
                "Some test IDs have no global FC probability."
            )
        global_fc_prob = test_global["prob_FC"].to_numpy(float)
    else:
        if len(global_pred) != len(test_info):
            raise ValueError(
                "Global prediction file cannot be aligned."
            )
        global_fc_prob = global_pred["prob_FC"].to_numpy(float)

    return (
        train_info, test_info,
        y_train, y_test,
        train_prob8, test_prob8,
        global_fc_prob, selected
    )


# ---------------------------------------------------------------------
# 5. GLOBAL CONTEXT ENCODING
# ---------------------------------------------------------------------

def make_global_context(train_info, test_info):
    """
    Uses the same conceptual one-hot context as Script 19.

    IMPORTANT:
    We fit category columns on the GLOBAL training partition and then subset
    rows for each scenario. This avoids changing context dimensionality from
    scenario to scenario and preserves the architecture meaning.
    """
    lf_train = pd.get_dummies(
        train_info[LANDFORM_COL].fillna("Missing").astype(str),
        prefix="topo",
        dtype=float,
    )
    lf_test = pd.get_dummies(
        test_info[LANDFORM_COL].fillna("Missing").astype(str),
        prefix="topo",
        dtype=float,
    )
    lf_test = lf_test.reindex(
        columns=lf_train.columns,
        fill_value=0.0
    )

    vg_train = pd.get_dummies(
        train_info[VEGETATION_COL].fillna("Missing").astype(str),
        prefix="veg",
        dtype=float,
    )
    vg_test = pd.get_dummies(
        test_info[VEGETATION_COL].fillna("Missing").astype(str),
        prefix="veg",
        dtype=float,
    )
    vg_test = vg_test.reindex(
        columns=vg_train.columns,
        fill_value=0.0
    )

    full_train = np.hstack([
        lf_train.to_numpy(np.float32),
        vg_train.to_numpy(np.float32)
    ]).astype(np.float32)

    full_test = np.hstack([
        lf_test.to_numpy(np.float32),
        vg_test.to_numpy(np.float32)
    ]).astype(np.float32)

    names = list(lf_train.columns) + list(vg_train.columns)

    return full_train, full_test, names


# ---------------------------------------------------------------------
# 6. FIT ONE SCENARIO-SPECIFIC META-ENSEMBLE
# ---------------------------------------------------------------------

def fit_scenario_model(
    scenario,
    scenario_type,
    train_mask,
    test_mask,
    train_prob8,
    test_prob8,
    full_context_train,
    full_context_test,
    y_train,
    y_test,
    selected,
    seed,
):
    p_train = train_prob8[train_mask]
    p_test = test_prob8[test_mask]

    c_train = full_context_train[train_mask]
    c_test = full_context_test[test_mask]

    ys_train = y_train[train_mask]
    ys_test = y_test[test_mask]

    if len(np.unique(ys_train)) < 2:
        raise ValueError(
            f"{scenario}: scenario training data contain only one class."
        )

    # Keep the architecture/hyperparameters fixed from Script 19.
    hidden = parse_hidden(selected["hidden_units"])
    dropout = float(selected["dropout"])
    l2_value = float(selected["l2"])
    learning_rate = float(selected["learning_rate"])
    batch_size = int(selected["batch_size"])

    # --------------------------------------------------------------
    # Internal split ONLY to determine a defensible number of epochs
    # for this scenario. Architecture is NOT retuned.
    # --------------------------------------------------------------
    idx = np.arange(len(ys_train))

    # If a very small scenario cannot support a normal stratified split,
    # use the global selected median epoch instead.
    counts = np.bincount(ys_train, minlength=2)
    can_validate = (
        len(ys_train) >= 100 and
        counts.min() >= 10
    )

    if can_validate:
        idx_fit, idx_val = train_test_split(
            idx,
            test_size=SCENARIO_VALID_SIZE,
            random_state=seed,
            stratify=ys_train,
        )

        temp_model = build_original_attention(
            n_models=len(BASE_MODELS),
            n_context=c_train.shape[1],
            hidden_units=hidden,
            dropout=dropout,
            l2_value=l2_value,
            learning_rate=learning_rate,
            seed=seed,
        )

        cb = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=PATIENCE,
            min_delta=MIN_DELTA,
            restore_best_weights=True,
            verbose=0,
        )

        hist = temp_model.fit(
            pack_inputs(
                p_train[idx_fit],
                c_train[idx_fit]
            ),
            ys_train[idx_fit],
            validation_data=(
                pack_inputs(
                    p_train[idx_val],
                    c_train[idx_val]
                ),
                ys_train[idx_val],
            ),
            epochs=MAX_EPOCHS,
            batch_size=batch_size,
            shuffle=True,
            callbacks=[cb],
            verbose=0,
        )

        best_epoch = int(
            np.argmin(hist.history["val_loss"]) + 1
        )

        # Save scenario internal-validation history.
        pd.DataFrame(hist.history).to_csv(
            OUT_DIR / f"history_internal_{scenario}.csv",
            index=False,
        )

        del temp_model
        tf.keras.backend.clear_session()

    else:
        best_epoch = int(
            round(float(selected["median_best_epoch"]))
        )

    best_epoch = max(5, best_epoch)

    # --------------------------------------------------------------
    # Final scenario model: ALL scenario training observations.
    # --------------------------------------------------------------
    final_model = build_original_attention(
        n_models=len(BASE_MODELS),
        n_context=c_train.shape[1],
        hidden_units=hidden,
        dropout=dropout,
        l2_value=l2_value,
        learning_rate=learning_rate,
        seed=seed + 1000,
    )

    final_hist = final_model.fit(
        pack_inputs(p_train, c_train),
        ys_train,
        epochs=best_epoch,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
    )

    pred2 = final_model.predict(
        pack_inputs(p_test, c_test),
        verbose=0,
    )
    prob = pred2[:, 1]

    # Attention weights on scenario test observations.
    attention_model = Model(
        inputs=final_model.inputs,
        outputs=final_model.get_layer(
            "attention_weights"
        ).output,
    )
    attention_weights = attention_model.predict(
        pack_inputs(p_test, c_test),
        verbose=0,
    )

    # Save final model and history.
    final_model.save(
        OUT_DIR / f"scenario_model_{scenario}.keras"
    )

    pd.DataFrame(final_hist.history).to_csv(
        OUT_DIR / f"history_final_{scenario}.csv",
        index=False,
    )

    return {
        "scenario": scenario,
        "scenario_type": scenario_type,
        "prob": prob,
        "attention": attention_weights,
        "best_epoch": best_epoch,
        "n_train": int(len(ys_train)),
        "n_test": int(len(ys_test)),
        "train_class1": int(np.sum(ys_train == 1)),
        "test_class1": int(np.sum(ys_test == 1)),
    }


# ---------------------------------------------------------------------
# 7. PAIRED STRATIFIED BOOTSTRAP
# ---------------------------------------------------------------------

def paired_bootstrap(
    y,
    scenario_prob,
    global_prob,
    n_boot=N_BOOTSTRAP,
    seed=42,
):
    y = np.asarray(y, int)
    scenario_prob = np.asarray(scenario_prob, float)
    global_prob = np.asarray(global_prob, float)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]

    if (
        len(idx0) < MIN_CLASS_N_FOR_BOOTSTRAP or
        len(idx1) < MIN_CLASS_N_FOR_BOOTSTRAP
    ):
        return pd.DataFrame([
            {
                "metric": metric,
                "mean_difference": np.nan,
                "ci_2.5": np.nan,
                "ci_97.5": np.nan,
                "bootstrap_valid": False,
                "reason": "Too few observations in one class",
            }
            for metric in ["auc", "f1_macro", "mcc"]
        ])

    rng = np.random.default_rng(seed)

    diffs = {
        "auc": [],
        "f1_macro": [],
        "mcc": [],
    }

    for _ in range(n_boot):
        s0 = rng.choice(
            idx0, size=len(idx0), replace=True
        )
        s1 = rng.choice(
            idx1, size=len(idx1), replace=True
        )

        sample = np.concatenate([s0, s1])
        rng.shuffle(sample)

        ms = calculate_metrics(
            y[sample],
            scenario_prob[sample]
        )
        mg = calculate_metrics(
            y[sample],
            global_prob[sample]
        )

        for metric in diffs:
            diffs[metric].append(
                ms[metric] - mg[metric]
            )

    rows = []
    for metric, values in diffs.items():
        values = np.asarray(values, float)
        rows.append({
            "metric": metric,
            "mean_difference": float(np.mean(values)),
            "ci_2.5": float(np.quantile(values, 0.025)),
            "ci_97.5": float(np.quantile(values, 0.975)),
            "bootstrap_valid": True,
            "reason": "",
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------

def main():
    set_seed(RANDOM_STATE)

    (
        train_info, test_info,
        y_train, y_test,
        train_prob8, test_prob8,
        global_fc_prob, selected
    ) = load_data()

    full_context_train, full_context_test, context_names = (
        make_global_context(train_info, test_info)
    )

    with open(
        OUT_DIR / "00_run_configuration.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "selected_architecture_from_script19": selected,
                "base_models": list(BASE_MODELS),
                "context_columns": context_names,
                "landform_scenarios": LANDFORM_SCENARIOS,
                "vegetation_scenarios": VEGETATION_SCENARIOS,
                "n_bootstrap": N_BOOTSTRAP,
                "interpretation": (
                    "Scenario-specific meta-ensemble specialisation; "
                    "global base learners are not retrained by scenario."
                ),
            },
            f,
            indent=2,
        )

    scenarios = (
        [("landform", LANDFORM_COL, s)
         for s in LANDFORM_SCENARIOS]
        +
        [("vegetation", VEGETATION_COL, s)
         for s in VEGETATION_SCENARIOS]
    )

    comparison_rows = []
    prediction_rows = []
    attention_rows = []
    bootstrap_rows = []

    print("=" * 78)
    print("SCENARIO-SPECIFIC ATTENTION ENSEMBLE VALIDATION")
    print("=" * 78)
    print("Architecture is fixed from Script 19.")
    print("The held-out test observations are not used for training/tuning.")
    print("\nSelected architecture:")
    print(selected)

    for scenario_number, (
        scenario_type, column, scenario
    ) in enumerate(scenarios, start=1):

        print("\n" + "-" * 78)
        print(
            f"[{scenario_number}/{len(scenarios)}] "
            f"{scenario_type.upper()}: {scenario}"
        )

        train_values = (
            train_info[column]
            .fillna("Missing")
            .astype(str)
        )
        test_values = (
            test_info[column]
            .fillna("Missing")
            .astype(str)
        )

        train_mask = (
            train_values == scenario
        ).to_numpy()

        test_mask = (
            test_values == scenario
        ).to_numpy()

        n_train = int(train_mask.sum())
        n_test = int(test_mask.sum())

        print(
            f"Train n={n_train:,}; "
            f"Test n={n_test:,}"
        )

        if n_train == 0 or n_test == 0:
            print(
                f"SKIPPED: no train/test observations for {scenario}."
            )
            continue

        result = fit_scenario_model(
            scenario=scenario,
            scenario_type=scenario_type,
            train_mask=train_mask,
            test_mask=test_mask,
            train_prob8=train_prob8,
            test_prob8=test_prob8,
            full_context_train=full_context_train,
            full_context_test=full_context_test,
            y_train=y_train,
            y_test=y_test,
            selected=selected,
            seed=RANDOM_STATE + scenario_number * 100,
        )

        ys_test = y_test[test_mask]
        scenario_prob = result["prob"]
        global_prob = global_fc_prob[test_mask]

        scenario_metrics = calculate_metrics(
            ys_test,
            scenario_prob
        )
        global_metrics = calculate_metrics(
            ys_test,
            global_prob
        )

        row = {
            "scenario_type": scenario_type,
            "scenario": scenario,
            "n_train": result["n_train"],
            "n_test": result["n_test"],
            "n_train_class1": result["train_class1"],
            "n_test_class1": result["test_class1"],
            "scenario_best_epoch": result["best_epoch"],
            "small_test_sample_warning": (
                result["n_test"] < MIN_TEST_N_WARNING
            ),

            "global_FC_accuracy": global_metrics["accuracy"],
            "scenario_ensemble_accuracy": scenario_metrics["accuracy"],
            "delta_accuracy": (
                scenario_metrics["accuracy"] -
                global_metrics["accuracy"]
            ),

            "global_FC_auc": global_metrics["auc"],
            "scenario_ensemble_auc": scenario_metrics["auc"],
            "delta_auc": (
                scenario_metrics["auc"] -
                global_metrics["auc"]
            ),

            "global_FC_f1_class1": global_metrics["f1_class1"],
            "scenario_ensemble_f1_class1": scenario_metrics["f1_class1"],
            "delta_f1_class1": (
                scenario_metrics["f1_class1"] -
                global_metrics["f1_class1"]
            ),

            "global_FC_f1_macro": global_metrics["f1_macro"],
            "scenario_ensemble_f1_macro": scenario_metrics["f1_macro"],
            "delta_f1_macro": (
                scenario_metrics["f1_macro"] -
                global_metrics["f1_macro"]
            ),

            "global_FC_mcc": global_metrics["mcc"],
            "scenario_ensemble_mcc": scenario_metrics["mcc"],
            "delta_mcc": (
                scenario_metrics["mcc"] -
                global_metrics["mcc"]
            ),
        }

        comparison_rows.append(row)

        print(
            f"Global FC AUC={global_metrics['auc']:.6f} | "
            f"Scenario-specific AUC={scenario_metrics['auc']:.6f} | "
            f"Delta={row['delta_auc']:+.6f}"
        )
        print(
            f"Global FC macro-F1={global_metrics['f1_macro']:.6f} | "
            f"Scenario-specific macro-F1={scenario_metrics['f1_macro']:.6f} | "
            f"Delta={row['delta_f1_macro']:+.6f}"
        )

        # ----------------------------------------------------------
        # Save observation-level predictions.
        # ----------------------------------------------------------
        scenario_test_info = test_info.loc[
            test_mask,
            [ID_COL, TARGET, LANDFORM_COL, VEGETATION_COL]
        ].reset_index(drop=True)

        pred_df = scenario_test_info.copy()
        pred_df["scenario_type"] = scenario_type
        pred_df["scenario"] = scenario
        pred_df["global_FC_probability"] = global_prob
        pred_df["scenario_ensemble_probability"] = scenario_prob
        pred_df["global_FC_prediction"] = (
            global_prob >= 0.5
        ).astype(int)
        pred_df["scenario_ensemble_prediction"] = (
            scenario_prob >= 0.5
        ).astype(int)

        prediction_rows.append(pred_df)

        # ----------------------------------------------------------
        # Attention weights.
        # ----------------------------------------------------------
        att_df = pd.DataFrame(
            result["attention"],
            columns=[
                f"attention_{m}"
                for m in BASE_MODELS
            ]
        )
        att_df.insert(0, "scenario", scenario)
        att_df.insert(0, "scenario_type", scenario_type)
        att_df.insert(
            0, ID_COL,
            scenario_test_info[ID_COL].values
        )
        attention_rows.append(att_df)

        # ----------------------------------------------------------
        # Paired bootstrap.
        # ----------------------------------------------------------
        boot = paired_bootstrap(
            ys_test,
            scenario_prob,
            global_prob,
            n_boot=N_BOOTSTRAP,
            seed=RANDOM_STATE + 5000 + scenario_number,
        )
        boot.insert(0, "scenario", scenario)
        boot.insert(0, "scenario_type", scenario_type)
        boot.insert(2, "n_test", len(ys_test))
        bootstrap_rows.append(boot)

    # -----------------------------------------------------------------
    # SAVE ALL RESULTS
    # -----------------------------------------------------------------

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(
        OUT_DIR / "01_scenario_specific_vs_global_FC_metrics.csv",
        index=False,
    )

    if prediction_rows:
        pd.concat(
            prediction_rows,
            ignore_index=True
        ).to_csv(
            OUT_DIR / "02_scenario_test_predictions.csv",
            index=False,
        )

    if attention_rows:
        all_attention = pd.concat(
            attention_rows,
            ignore_index=True
        )
        all_attention.to_csv(
            OUT_DIR / "03_scenario_attention_weights.csv",
            index=False,
        )

        attention_summary = (
            all_attention
            .groupby(
                ["scenario_type", "scenario"],
                as_index=False
            )[
                [f"attention_{m}" for m in BASE_MODELS]
            ]
            .agg(["mean", "std"])
        )

        # Flatten MultiIndex columns.
        attention_summary.columns = [
            "_".join(
                [str(x) for x in col if str(x) != ""]
            ).rstrip("_")
            if isinstance(col, tuple)
            else str(col)
            for col in attention_summary.columns
        ]

        attention_summary.to_csv(
            OUT_DIR / "04_scenario_attention_summary.csv",
            index=False,
        )

    if bootstrap_rows:
        bootstrap = pd.concat(
            bootstrap_rows,
            ignore_index=True
        )
        bootstrap.to_csv(
            OUT_DIR / "05_bootstrap_scenario_vs_global_FC.csv",
            index=False,
        )

    # -----------------------------------------------------------------
    # Compact thesis-ready table.
    # -----------------------------------------------------------------
    if not comparison.empty:
        thesis_table = comparison[
            [
                "scenario_type",
                "scenario",
                "n_train",
                "n_test",
                "global_FC_auc",
                "scenario_ensemble_auc",
                "delta_auc",
                "global_FC_f1_macro",
                "scenario_ensemble_f1_macro",
                "delta_f1_macro",
                "global_FC_mcc",
                "scenario_ensemble_mcc",
                "delta_mcc",
                "small_test_sample_warning",
            ]
        ].copy()

        thesis_table.to_csv(
            OUT_DIR / "06_thesis_ready_scenario_comparison.csv",
            index=False,
        )

        # -----------------------------------------------------------------
        # Human-readable summary.
        # -----------------------------------------------------------------
        lines = [
            "Appendix A scenario-specific ensemble validation",
            "=" * 50,
            "",
            "Question:",
            "Does specialising the attention meta-ensemble to an individual",
            "landform or vegetation scenario improve prediction relative to",
            "the global FC ensemble evaluated in the same test scenario?",
            "",
            "IMPORTANT:",
            "The RF/XGB/SVM/MLP base learners are NOT retrained by scenario.",
            "This isolates scenario-specific META-ENSEMBLE specialisation.",
            "",
            "Architecture:",
            f"- hidden units: {selected['hidden_units']}",
            f"- dropout: {selected['dropout']}",
            f"- L2: {selected['l2']}",
            f"- learning rate: {selected['learning_rate']}",
            f"- batch size: {selected['batch_size']}",
            "",
            "Results:",
        ]

        for _, r in comparison.iterrows():
            lines.extend([
                (
                    f"- {r['scenario']} ({r['scenario_type']}, "
                    f"test n={int(r['n_test'])}): "
                    f"global FC AUC={r['global_FC_auc']:.6f}; "
                    f"scenario-specific AUC="
                    f"{r['scenario_ensemble_auc']:.6f}; "
                    f"delta={r['delta_auc']:+.6f}"
                )
            ])

        lines.extend([
            "",
            "Interpret differences together with paired bootstrap CIs.",
            "Small-sample vegetation scenarios should be interpreted cautiously.",
        ])

        (OUT_DIR / "RUN_SUMMARY.txt").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

    print("\n" + "=" * 78)
    print("COMPLETE")
    print("=" * 78)

    if not comparison.empty:
        print(
            comparison[
                [
                    "scenario_type",
                    "scenario",
                    "n_train",
                    "n_test",
                    "global_FC_auc",
                    "scenario_ensemble_auc",
                    "delta_auc",
                    "global_FC_f1_macro",
                    "scenario_ensemble_f1_macro",
                ]
            ].to_string(index=False)
        )

    print(f"\nOutputs saved to:\n{OUT_DIR}")


if __name__ == "__main__":
    main()
