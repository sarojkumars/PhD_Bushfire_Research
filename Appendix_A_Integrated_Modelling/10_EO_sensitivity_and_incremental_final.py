"""Appendix A — Additional fire-severity modelling analyses.

EO severity sensitivity, paired bootstrap comparisons, landform transferability
and environmental-domain comparisons.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)
from xgboost import XGBClassifier


# Settings

SCRIPT_DIR = Path(__file__).resolve().parent
from appendix_config import DATA_FILE, VAL_DIR, ADD_DIR
VALIDATION_DIR = VAL_DIR
OUTPUT_DIR = ADD_DIR

ID_COL = "pt_id"
MAIN_TARGET = "SNIC2clsN"
LANDFORM_COL = "LF4Cls"

RANDOM_STATE = 42
N_BOOTSTRAP = 2000
THRESHOLD = 0.5

N_JOBS = -1


EO_TARGETS = {
    "dNBR": "dNBR2ClsN",
    "RdNBR": "RdNBR2ClsN",
    "RBR": "RBR2ClsN",
    "SNIC": "SNIC2clsN",
}


# Predictor domains

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

DOMAIN_ORDER = ["M", "T", "FV", "M+T", "M+FV", "T+FV", "M+T+FV"]


# Functions

def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_columns(df: pd.DataFrame, cols, label="required"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing {label} columns: {missing}")


def binary_metrics(y_true, prob, threshold=THRESHOLD):
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float).reshape(-1)
    pred = (prob >= threshold).astype(int)

    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, prob)
    else:
        auc = np.nan

    tn, fp, fn, tp = confusion_matrix(
        y_true, pred, labels=[0, 1]
    ).ravel()

    return {
        "n": len(y_true),
        "prevalence_class1": y_true.mean(),
        "auc": auc,
        "accuracy": accuracy_score(y_true, pred),
        "precision_class1": precision_score(
            y_true, pred, zero_division=0
        ),
        "recall_class1": recall_score(
            y_true, pred, zero_division=0
        ),
        "f1_class1": f1_score(
            y_true, pred, zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, pred, average="macro", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, pred, average="weighted", zero_division=0
        ),
        "mcc": matthews_corrcoef(y_true, pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_main_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_FILE}\n"
            "Edit DATA_FILE near the top of this script."
        )

    split_file = VALIDATION_DIR / "01_fixed_split_ids.csv"
    if not split_file.exists():
        raise FileNotFoundError(
            f"Fixed split file not found:\n{split_file}"
        )

    needed = (
        [ID_COL, MAIN_TARGET, LANDFORM_COL]
        + list(EO_TARGETS.values())
        + ALL_PREDICTORS
    )

    df = pd.read_csv(DATA_FILE, usecols=lambda c: c in set(needed))
    split = pd.read_csv(split_file)

    ensure_columns(df, needed)
    ensure_columns(split, [ID_COL, "split"])

    df = df.merge(
        split[[ID_COL, "split"]],
        on=ID_COL,
        how="inner",
        validate="one_to_one",
    )

    if len(df) != len(split):
        raise ValueError(
            "Not every fixed-split pt_id was recovered from the source CSV."
        )

    return df


def get_reference_xgb_params():
    f = VALIDATION_DIR / "10_environmental_domain_ablation_metrics.csv"

    defaults = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
    }

    if not f.exists():
        print("WARNING: domain-ablation table not found; using fixed XGB defaults.")
        return defaults

    tab = pd.read_csv(f)
    row = tab.loc[tab["domain_set"] == "M+T+FV"]

    if row.empty or "best_params" not in row.columns:
        print("WARNING: M+T+FV tuned parameters not found; using defaults.")
        return defaults

    raw = json.loads(row.iloc[0]["best_params"])

    params = {}
    for k, v in raw.items():
        params[k.replace("model__", "")] = v

    print("\nReference XGB parameters recovered from main validation:")
    print(params)
    return params


def make_xgb(params):
    model_params = dict(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=N_JOBS,
    )
    model_params.update(params)

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(**model_params)),
        ]
    )


def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


# EO severity sensitivity

def run_eo_severity_sensitivity(df, xgb_params):
    print("\n" + "=" * 78)
    print("EXPERIMENT A: EO SEVERITY-REPRESENTATION SENSITIVITY")
    print("=" * 78)

    rows = []

    for response_name, target_col in EO_TARGETS.items():
        print(f"\nEO response: {response_name} [{target_col}]")

        sub = df.dropna(subset=[target_col]).copy()

        vals = set(pd.unique(sub[target_col]))
        if not vals.issubset({0, 1}):
            print(
                f"SKIPPING {response_name}: {target_col} is not binary. "
                f"Observed values: {sorted(vals)}"
            )
            continue

        train = sub.loc[sub["split"] == "TRAIN"].copy()
        test = sub.loc[sub["split"] == "TEST"].copy()

        y_train = train[target_col].astype(int)
        y_test = test[target_col].astype(int)

        if y_train.nunique() < 2 or y_test.nunique() < 2:
            print(f"SKIPPING {response_name}: both classes not present.")
            continue

        for domain in DOMAIN_ORDER:
            predictors = DOMAIN_SETS[domain]
            print(f"  {domain:7s} | train={len(train):,} test={len(test):,}")

            model = make_xgb(xgb_params)
            model.fit(train[predictors], y_train)
            prob = model.predict_proba(test[predictors])[:, 1]

            m = binary_metrics(y_test, prob)
            m.update(
                {
                    "severity_representation": response_name,
                    "target_column": target_col,
                    "domain_set": domain,
                    "n_predictors": len(predictors),
                }
            )
            rows.append(m)

    out = pd.DataFrame(rows)
    out.to_csv(
        OUTPUT_DIR / "A1_EO_severity_domain_sensitivity_metrics.csv",
        index=False,
    )

    prevalence_rows = []
    for response_name, target_col in EO_TARGETS.items():
        if target_col not in df.columns:
            continue
        for split_name in ["TRAIN", "TEST"]:
            s = df.loc[df["split"] == split_name, target_col].dropna()
            if len(s) and set(s.unique()).issubset({0, 1}):
                prevalence_rows.append(
                    {
                        "severity_representation": response_name,
                        "target_column": target_col,
                        "split": split_name,
                        "n": len(s),
                        "class1_n": int(s.sum()),
                        "class1_prevalence": s.mean(),
                    }
                )

    pd.DataFrame(prevalence_rows).to_csv(
        OUTPUT_DIR / "A2_EO_severity_response_prevalence.csv",
        index=False,
    )

    plot_eo_heatmap(out)
    return out


def plot_eo_heatmap(results):
    if results.empty:
        return

    pivot = results.pivot(
        index="severity_representation",
        columns="domain_set",
        values="auc",
    )

    response_order = [
        x for x in EO_TARGETS.keys() if x in pivot.index
    ]
    domain_order = [x for x in DOMAIN_ORDER if x in pivot.columns]

    pivot = pivot.reindex(
        index=response_order,
        columns=domain_order,
    )

    arr = pivot.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    im = ax.imshow(arr, aspect="auto")

    ax.set_xticks(np.arange(len(domain_order)))
    ax.set_xticklabels(domain_order)
    ax.set_yticks(np.arange(len(response_order)))
    ax.set_yticklabels(response_order)

    ax.set_xlabel("Environmental predictor domain")
    ax.set_ylabel("EO severity representation")
    ax.set_title(
        "Sensitivity of predictive discrimination to EO severity representation"
    )

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isfinite(arr[i, j]):
                ax.text(
                    j, i, f"{arr[i, j]:.3f}",
                    ha="center", va="center"
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Test AUC")

    save_figure(fig, "Figure_A_EO_severity_domain_AUC_heatmap")


# Paired bootstrap comparisons

def paired_bootstrap_difference(
    y,
    prob_a,
    prob_b,
    metric,
    n_boot=N_BOOTSTRAP,
    seed=RANDOM_STATE,
):
    y = np.asarray(y).astype(int)
    a = np.asarray(prob_a, dtype=float)
    b = np.asarray(prob_b, dtype=float)

    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]

        if metric == "auc" and np.unique(yy).size < 2:
            continue

        aa = a[idx]
        bb = b[idx]

        if metric == "auc":
            va = roc_auc_score(yy, aa)
            vb = roc_auc_score(yy, bb)

        elif metric == "f1_macro":
            va = f1_score(
                yy, (aa >= THRESHOLD).astype(int),
                average="macro", zero_division=0
            )
            vb = f1_score(
                yy, (bb >= THRESHOLD).astype(int),
                average="macro", zero_division=0
            )

        elif metric == "mcc":
            va = matthews_corrcoef(
                yy, (aa >= THRESHOLD).astype(int)
            )
            vb = matthews_corrcoef(
                yy, (bb >= THRESHOLD).astype(int)
            )
        else:
            raise ValueError(metric)

        diffs.append(vb - va)

    diffs = np.asarray(diffs)

    return {
        "observed_difference": None,  # filled below
        "bootstrap_mean_difference": np.mean(diffs),
        "ci_2.5": np.quantile(diffs, 0.025),
        "ci_97.5": np.quantile(diffs, 0.975),
        "bootstrap_p_difference_gt_0": np.mean(diffs > 0),
        "n_bootstrap_valid": len(diffs),
    }


def metric_value(y, p, metric):
    if metric == "auc":
        return roc_auc_score(y, p)
    pred = (np.asarray(p) >= THRESHOLD).astype(int)
    if metric == "f1_macro":
        return f1_score(y, pred, average="macro", zero_division=0)
    if metric == "mcc":
        return matthews_corrcoef(y, pred)
    raise ValueError(metric)


def run_bootstrap_uncertainty():
    print("\n" + "=" * 78)
    print("EXPERIMENT B: PAIRED BOOTSTRAP UNCERTAINTY")
    print("=" * 78)

    context_file = VALIDATION_DIR / "06_context_ablation_predictions.csv"
    base_file = VALIDATION_DIR / "04_test_base_probabilities.csv"

    context = pd.read_csv(context_file)
    base = pd.read_csv(base_file)

    if len(context) != len(base):
        raise ValueError(
            "04_test_base_probabilities.csv and "
            "06_context_ablation_predictions.csv have different lengths."
        )

    y = context[MAIN_TARGET].to_numpy().astype(int)

    xgb_prob = base["XGB"].to_numpy()

    comparisons = {
        "FC - CF": (
            context["prob_CF"].to_numpy(),
            context["prob_FC"].to_numpy(),
        ),
        "TC - CF": (
            context["prob_CF"].to_numpy(),
            context["prob_TC"].to_numpy(),
        ),
        "VC - CF": (
            context["prob_CF"].to_numpy(),
            context["prob_VC"].to_numpy(),
        ),
        "CF ensemble - XGB": (
            xgb_prob,
            context["prob_CF"].to_numpy(),
        ),
        "FC ensemble - XGB": (
            xgb_prob,
            context["prob_FC"].to_numpy(),
        ),
    }

    rows = []

    for comparison, (a, b) in comparisons.items():
        print(f"\n{comparison}")

        for metric in ["auc", "f1_macro", "mcc"]:
            result = paired_bootstrap_difference(
                y, a, b, metric
            )

            observed = metric_value(y, b, metric) - metric_value(y, a, metric)
            result["observed_difference"] = observed
            result["comparison"] = comparison
            result["metric"] = metric
            rows.append(result)

            print(
                f"  {metric:9s}: Δ={observed:+.5f}, "
                f"95% CI [{result['ci_2.5']:+.5f}, "
                f"{result['ci_97.5']:+.5f}]"
            )

    out = pd.DataFrame(rows)
    out.to_csv(
        OUTPUT_DIR / "B1_paired_bootstrap_model_differences.csv",
        index=False,
    )

    plot_bootstrap_forest(out)
    return out


def plot_bootstrap_forest(results):
    d = results.loc[results["metric"] == "auc"].copy()
    d = d.iloc[::-1].reset_index(drop=True)

    y_pos = np.arange(len(d))
    x = d["observed_difference"].to_numpy()
    lower = x - d["ci_2.5"].to_numpy()
    upper = d["ci_97.5"].to_numpy() - x

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.errorbar(
        x,
        y_pos,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=4,
    )
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(d["comparison"])
    ax.set_xlabel("Difference in test AUC (second model − first model)")
    ax.set_title("Paired bootstrap uncertainty in model differences")

    save_figure(fig, "Figure_B_paired_bootstrap_AUC_differences")


# Landform transferability

def run_landform_transferability(df, xgb_params):
    print("\n" + "=" * 78)
    print("EXPERIMENT C: LEAVE-ONE-LANDFORM-OUT TRANSFERABILITY")
    print("=" * 78)

    sub = df.dropna(
        subset=[MAIN_TARGET, LANDFORM_COL]
    ).copy()

    landforms = sorted(sub[LANDFORM_COL].astype(str).unique())
    rows = []
    prediction_rows = []

    for landform in landforms:
        print(f"\nHeld-out landform: {landform}")

        test_context = sub.loc[
            (sub["split"] == "TEST")
            & (sub[LANDFORM_COL].astype(str) == landform)
        ].copy()

        train_within = sub.loc[
            (sub["split"] == "TRAIN")
            & (sub[LANDFORM_COL].astype(str) == landform)
        ].copy()

        train_other = sub.loc[
            (sub["split"] == "TRAIN")
            & (sub[LANDFORM_COL].astype(str) != landform)
        ].copy()

        if (
            len(test_context) == 0
            or train_within[MAIN_TARGET].nunique() < 2
            or train_other[MAIN_TARGET].nunique() < 2
        ):
            print("  Skipping due to insufficient classes/observations.")
            continue

        y_test = test_context[MAIN_TARGET].astype(int)

        within_model = make_xgb(xgb_params)
        within_model.fit(
            train_within[ALL_PREDICTORS],
            train_within[MAIN_TARGET].astype(int),
        )
        p_within = within_model.predict_proba(
            test_context[ALL_PREDICTORS]
        )[:, 1]

        m_within = binary_metrics(y_test, p_within)
        m_within.update(
            {
                "landform": landform,
                "training_condition": "within_context",
                "n_train": len(train_within),
            }
        )
        rows.append(m_within)

        transfer_model = make_xgb(xgb_params)
        transfer_model.fit(
            train_other[ALL_PREDICTORS],
            train_other[MAIN_TARGET].astype(int),
        )
        p_transfer = transfer_model.predict_proba(
            test_context[ALL_PREDICTORS]
        )[:, 1]

        m_transfer = binary_metrics(y_test, p_transfer)
        m_transfer.update(
            {
                "landform": landform,
                "training_condition": "unseen_context",
                "n_train": len(train_other),
            }
        )
        rows.append(m_transfer)

        pred = pd.DataFrame(
            {
                ID_COL: test_context[ID_COL].values,
                "landform": landform,
                "y_true": y_test.values,
                "prob_within_context": p_within,
                "prob_unseen_context": p_transfer,
            }
        )
        prediction_rows.append(pred)

        print(
            f"  Within-context AUC = {m_within['auc']:.3f} | "
            f"Unseen-context AUC = {m_transfer['auc']:.3f} | "
            f"Δ = {m_transfer['auc'] - m_within['auc']:+.3f}"
        )

    metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)

    metrics.to_csv(
        OUTPUT_DIR / "C1_landform_transferability_metrics.csv",
        index=False,
    )
    predictions.to_csv(
        OUTPUT_DIR / "C2_landform_transferability_predictions.csv",
        index=False,
    )

    gap_rows = []

    for landform, g in predictions.groupby("landform"):
        y = g["y_true"].to_numpy()
        within = g["prob_within_context"].to_numpy()
        unseen = g["prob_unseen_context"].to_numpy()

        for metric in ["auc", "f1_macro", "mcc"]:
            res = paired_bootstrap_difference(
                y, within, unseen, metric,
                seed=RANDOM_STATE + sum(ord(c) for c in str(landform)),
            )
            res["observed_difference"] = (
                metric_value(y, unseen, metric)
                - metric_value(y, within, metric)
            )
            res["landform"] = landform
            res["metric"] = metric
            gap_rows.append(res)

    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(
        OUTPUT_DIR / "C3_landform_transferability_bootstrap_gaps.csv",
        index=False,
    )

    plot_landform_transfer(metrics)
    plot_landform_transfer_gap(gaps)

    return metrics, gaps


def plot_landform_transfer(metrics):
    pivot = metrics.pivot(
        index="landform",
        columns="training_condition",
        values="auc",
    )

    landforms = list(pivot.index)
    x = np.arange(len(landforms))

    fig, ax = plt.subplots(figsize=(9, 5.5))

    if "within_context" in pivot.columns:
        ax.scatter(
            x - 0.08,
            pivot["within_context"],
            label="Within-context training",
            marker="o",
        )

    if "unseen_context" in pivot.columns:
        ax.scatter(
            x + 0.08,
            pivot["unseen_context"],
            label="Landform unseen in training",
            marker="s",
        )

    if {"within_context", "unseen_context"}.issubset(pivot.columns):
        for i in range(len(landforms)):
            ax.plot(
                [x[i] - 0.08, x[i] + 0.08],
                [
                    pivot.iloc[i]["within_context"],
                    pivot.iloc[i]["unseen_context"],
                ],
                linewidth=1,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(landforms)
    ax.set_ylabel("Test AUC")
    ax.set_xlabel("Target landform")
    ax.set_title("Transferability to landform contexts absent from training")
    ax.legend()

    save_figure(fig, "Figure_C_landform_transferability_AUC")


def plot_landform_transfer_gap(gaps):
    d = gaps.loc[gaps["metric"] == "auc"].copy()
    d = d.sort_values("landform").reset_index(drop=True)

    x = d["observed_difference"].to_numpy()
    lower = x - d["ci_2.5"].to_numpy()
    upper = d["ci_97.5"].to_numpy() - x
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.errorbar(
        x,
        y,
        xerr=np.vstack([lower, upper]),
        fmt="o",
        capsize=4,
    )
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["landform"])
    ax.set_xlabel(
        "ΔAUC: unseen-context training − within-context training"
    )
    ax.set_title("Uncertainty in landform transferability gap")

    save_figure(fig, "Figure_C2_landform_transferability_gap")


# Environmental domain comparisons

def run_incremental_information():
    print("\n" + "=" * 78)
    print("EXPERIMENT D: INCREMENTAL ENVIRONMENTAL INFORMATION")
    print("=" * 78)

    f = VALIDATION_DIR / "10_environmental_domain_ablation_metrics.csv"
    tab = pd.read_csv(f).set_index("domain_set")

    required = set(DOMAIN_ORDER)
    missing = required.difference(tab.index)
    if missing:
        raise ValueError(f"Missing domain-ablation rows: {sorted(missing)}")

    comparisons = [
        ("Add T to M", "M", "M+T", "T"),
        ("Add FV to M", "M", "M+FV", "FV"),
        ("Add M to T", "T", "M+T", "M"),
        ("Add FV to T", "T", "T+FV", "FV"),
        ("Add M to FV", "FV", "M+FV", "M"),
        ("Add T to FV", "FV", "T+FV", "T"),
        ("Add FV to M+T", "M+T", "M+T+FV", "FV"),
        ("Add T to M+FV", "M+FV", "M+T+FV", "T"),
        ("Add M to T+FV", "T+FV", "M+T+FV", "M"),
    ]

    rows = []

    for label, base, expanded, added in comparisons:
        row = {
            "comparison": label,
            "base_domain_set": base,
            "expanded_domain_set": expanded,
            "added_domain": added,
        }

        for metric in ["auc", "f1_macro", "mcc", "accuracy"]:
            base_value = float(tab.loc[base, metric])
            expanded_value = float(tab.loc[expanded, metric])

            row[f"{metric}_base"] = base_value
            row[f"{metric}_expanded"] = expanded_value
            row[f"delta_{metric}"] = expanded_value - base_value

        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(
        OUTPUT_DIR / "D1_incremental_environmental_information.csv",
        index=False,
    )

    clean = tab.reset_index()[
        [
            "domain_set",
            "n_predictors",
            "auc",
            "f1_macro",
            "f1_weighted",
            "mcc",
            "accuracy",
        ]
    ].copy()

    clean["domain_set"] = pd.Categorical(
        clean["domain_set"],
        categories=DOMAIN_ORDER,
        ordered=True,
    )
    clean = clean.sort_values("domain_set")
    clean.to_csv(
        OUTPUT_DIR / "D2_environmental_domain_performance_clean.csv",
        index=False,
    )

    plot_incremental_information(out)
    plot_domain_performance(clean)

    return out


def plot_incremental_information(out):
    d = out.sort_values("delta_auc").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.5, 6))
    y = np.arange(len(d))
    ax.barh(y, d["delta_auc"])
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(d["comparison"])
    ax.set_xlabel("Increment in test AUC")
    ax.set_title(
        "Incremental predictive information from environmental domains"
    )

    for i, v in enumerate(d["delta_auc"]):
        ha = "left" if v >= 0 else "right"
        offset = 0.0005 if v >= 0 else -0.0005
        ax.text(v + offset, i, f"{v:+.3f}", va="center", ha=ha)

    save_figure(fig, "Figure_D_incremental_environmental_information")


def plot_domain_performance(clean):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(clean))
    ax.plot(x, clean["auc"], marker="o")
    ax.set_xticks(x)
    ax.set_xticklabels(clean["domain_set"])
    ax.set_xlabel("Environmental predictor domain")
    ax.set_ylabel("Test AUC")
    ax.set_title(
        "Standalone and integrated environmental predictive information"
    )

    for i, v in enumerate(clean["auc"]):
        ax.text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom")

    save_figure(fig, "Figure_D2_environmental_domain_AUC")


# Run analyses

def write_readme(eo, bootstrap, transfer, gaps, incremental, xgb_params):
    lines = [
        "APPENDIX A ADDITIONAL EXPERIMENTS — RUN SUMMARY",
        "=" * 58,
        "",
        "Reference XGB parameters:",
        json.dumps(xgb_params, indent=2),
        "",
        "A. EO severity-representation sensitivity",
        f"Rows in result table: {len(eo)}",
        "Interpretation: tests whether domain-level predictive conclusions are",
        "robust to alternative supplied binary EO severity representations.",
        "",
        "B. Paired bootstrap uncertainty",
        f"Bootstrap replicates requested: {N_BOOTSTRAP}",
        "Interpretation: quantifies uncertainty in CF/TC/VC/FC and ensemble-XGB",
        "differences using paired resampling of the same held-out observations.",
        "",
        "C. Leave-one-landform-out transferability",
        f"Landform metric rows: {len(transfer)}",
        "Interpretation: tests performance when the target landform was absent",
        "from model development, compared on the same landform-specific test set.",
        "",
        "D. Incremental environmental information",
        f"Increment comparisons: {len(incremental)}",
        "Interpretation: descriptive change in predictive performance when one",
        "environmental domain is added to another domain combination.",
        "",
        "CAUTIONS",
        "--------",
        "- Incremental AUC is predictive information, not causal effect.",
        "- Attention/model bootstrap comparisons do not prove equivalence when a",
        "  CI includes zero; describe the estimated difference and uncertainty.",
        "- LOLO is a transferability stress test; training sample sizes differ.",
        "- EO target comparisons must be interpreted alongside class prevalence.",
        "- MVClsN is intentionally excluded because it is four-class in the supplied",
        "  dataset; do not create a binary mapping without a documented thesis rule.",
        "- None of these analyses replaces spatial-block validation.",
    ]

    (OUTPUT_DIR / "README_interpretation_notes.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    warnings.filterwarnings("ignore")
    ensure_dirs()

    print("\nLoading fixed Appendix A data/split...")
    df = load_main_data()
    xgb_params = get_reference_xgb_params()

    eo = run_eo_severity_sensitivity(df, xgb_params)
    bootstrap = run_bootstrap_uncertainty()
    transfer, gaps = run_landform_transferability(df, xgb_params)
    incremental = run_incremental_information()

    write_readme(
        eo,
        bootstrap,
        transfer,
        gaps,
        incremental,
        xgb_params,
    )

    print("\n" + "=" * 78)
    print("ALL ADDITIONAL APPENDIX A EXPERIMENTS COMPLETE")
    print("=" * 78)
    print(f"\nOutputs saved to:\n{OUTPUT_DIR}")

    print("\nKey output tables:")
    print(" A1_EO_severity_domain_sensitivity_metrics.csv")
    print(" B1_paired_bootstrap_model_differences.csv")
    print(" C1_landform_transferability_metrics.csv")
    print(" C3_landform_transferability_bootstrap_gaps.csv")
    print(" D1_incremental_environmental_information.csv")

    print("\nKey figures:")
    print(" Figure_A_EO_severity_domain_AUC_heatmap")
    print(" Figure_B_paired_bootstrap_AUC_differences")
    print(" Figure_C_landform_transferability_AUC")
    print(" Figure_C2_landform_transferability_gap")
    print(" Figure_D_incremental_environmental_information")
    print(" Figure_D2_environmental_domain_AUC")


if __name__ == "__main__":
    main()
