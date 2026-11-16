import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from appendix_config import *


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

SHAP_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD FINAL FULL ENVIRONMENTAL XGB MODEL
# ============================================================

mf = (
    VAL_DIR
    / "domain_ablation_XGB_M_T_FV.joblib"
)

if not mf.exists():
    raise FileNotFoundError(
        "Run domain ablation first."
    )

print(
    f"Loading fitted XGB model:\n{mf}",
    flush=True
)

model = joblib.load(
    mf
)


# ============================================================
# LOAD DATA AND RECONSTRUCT FIXED TEST SET
# ============================================================

df = pd.read_csv(
    DATA_FILE
)

split_file = (
    VAL_DIR
    / "01_fixed_split_ids.csv"
)

if not split_file.exists():
    raise FileNotFoundError(
        f"Fixed split file not found: "
        f"{split_file}"
    )

sp = pd.read_csv(
    split_file
)

ids = (
    sp.loc[
        sp["split"] == "TEST",
        ID_COL
    ]
    .tolist()
)

test = (
    df
    .set_index(ID_COL)
    .loc[ids]
    .reset_index()
)

print(
    f"Fixed test observations: "
    f"{len(test):,}",
    flush=True
)


# ============================================================
# REPRODUCIBLE SHAP SAMPLE
# ============================================================

sample = (
    test
    .sample(
        n=min(
            5000,
            len(test)
        ),
        random_state=RANDOM_STATE
    )
    .copy()
)

X = sample[
    ALL_PREDICTORS
]

print(
    f"SHAP sample size: "
    f"{len(sample):,}",
    flush=True
)


# ============================================================
# APPLY FITTED PREPROCESSING
# ============================================================

Xt = (
    model
    .named_steps["imputer"]
    .transform(X)
)

Xt = (
    model
    .named_steps["scaler"]
    .transform(Xt)
)

xgb_model = (
    model
    .named_steps["model"]
)


# ============================================================
# NATIVE XGBOOST TREESHAP
#
# This avoids the SHAP 0.49.1 / XGBoost 3.2.0
# base_score parser incompatibility.
#
# pred_contribs returns:
#   n observations x (n predictors + 1)
#
# The final column is the expected-value/bias term and is
# therefore excluded from predictor attribution.
# ============================================================

print(
    "Calculating native XGBoost TreeSHAP "
    "contributions...",
    flush=True
)

booster = (
    xgb_model
    .get_booster()
)

dmatrix = xgb.DMatrix(
    Xt,
    feature_names=list(
        ALL_PREDICTORS
    )
)

contrib = booster.predict(
    dmatrix,
    pred_contribs=True
)

contrib = np.asarray(
    contrib
)

expected_columns = (
    len(ALL_PREDICTORS)
    + 1
)

if (
    contrib.ndim != 2
    or
    contrib.shape[1]
    != expected_columns
):
    raise RuntimeError(
        "Unexpected XGBoost SHAP contribution "
        f"shape: {contrib.shape}. "
        f"Expected "
        f"(n, {expected_columns})."
    )

# Predictor SHAP values
sv = contrib[
    :,
    :-1
]

# Bias / expected-value contribution
bias = contrib[
    :,
    -1
]


# ============================================================
# SAVE RAW SHAP RESULTS
# ============================================================

np.save(
    SHAP_DIR
    / "shap_values.npy",
    sv
)

np.save(
    SHAP_DIR
    / "shap_bias.npy",
    bias
)

sample[
    [
        ID_COL,
        TARGET,
        LANDFORM_COL,
        VEGETATION_COL
    ]
    +
    ALL_PREDICTORS
].to_csv(
    SHAP_DIR
    / "shap_sample.csv",
    index=False
)


# ============================================================
# GLOBAL FEATURE ATTRIBUTION
# ============================================================

ma = (
    np.abs(sv)
    .mean(axis=0)
)

imp = pd.DataFrame(
    {
        "feature":
            ALL_PREDICTORS,

        "mean_abs_shap":
            ma
    }
)


# ============================================================
# ENVIRONMENTAL DOMAIN MAPPING
# ============================================================

domain_map = {
    **{
        feature: "M"
        for feature
        in METEOROLOGY
    },

    **{
        feature: "T"
        for feature
        in TOPOGRAPHY
    },

    **{
        feature: "FV"
        for feature
        in FUEL_VEGETATION
    }
}

imp["domain"] = (
    imp["feature"]
    .map(domain_map)
)

if imp["domain"].isna().any():

    missing_domain = (
        imp.loc[
            imp["domain"].isna(),
            "feature"
        ]
        .tolist()
    )

    raise RuntimeError(
        "Predictors missing environmental "
        f"domain assignment: "
        f"{missing_domain}"
    )

imp = (
    imp
    .sort_values(
        "mean_abs_shap",
        ascending=False
    )
)

imp.to_csv(
    SHAP_DIR
    / "SHAP_global_importance.csv",
    index=False
)


# ============================================================
# NORMALISED DOMAIN ATTRIBUTION
# ============================================================

dom = (
    imp
    .groupby(
        "domain",
        as_index=False
    )["mean_abs_shap"]
    .sum()
)

total_abs_shap = (
    dom[
        "mean_abs_shap"
    ]
    .sum()
)

dom[
    "normalised_abs_shap"
] = (
    dom[
        "mean_abs_shap"
    ]
    /
    total_abs_shap
)

dom.to_csv(
    SHAP_DIR
    / "SHAP_domain_attribution.csv",
    index=False
)


# ============================================================
# CONTEXT-SPECIFIC ATTRIBUTION
#
# IMPORTANT:
# These are subgroup summaries of the SAME global model.
# Separate context-specific models are not fitted here.
# ============================================================

rows = []

for typ, col in [
    (
        "landform",
        LANDFORM_COL
    ),
    (
        "vegetation",
        VEGETATION_COL
    )
]:

    grouped = (
        sample
        .groupby(
            col,
            dropna=False
        )
        .indices
    )

    for level, pos in grouped.items():

        pos = np.asarray(
            pos,
            dtype=int
        )

        if len(pos) < 20:
            continue

        context_abs = (
            np.abs(
                sv[pos]
            )
            .mean(axis=0)
        )

        denominator = (
            context_abs.sum()
        )

        if denominator <= 0:
            continue

        for feature, value in zip(
            ALL_PREDICTORS,
            context_abs
        ):

            rows.append(
                {
                    "context_type":
                        typ,

                    "context_level":
                        level,

                    "n":
                        len(pos),

                    "feature":
                        feature,

                    "mean_abs_shap":
                        value,

                    "normalised_abs_shap":
                        value
                        /
                        denominator
                }
            )


context_imp = pd.DataFrame(
    rows
)

context_imp.to_csv(
    SHAP_DIR
    / "SHAP_context_importance.csv",
    index=False
)


# ============================================================
# TOP-20 GLOBAL ATTRIBUTION FIGURE
# ============================================================

top = (
    imp
    .head(20)
    .copy()
)

plt.figure(
    figsize=(8, 7)
)

plt.barh(
    top["feature"][::-1],
    top["mean_abs_shap"][::-1]
)

plt.xlabel(
    "Mean |SHAP value|"
)

plt.ylabel(
    "Predictor"
)

plt.title(
    "Environmental attribution in the "
    "full XGB severity model"
)

plt.tight_layout()

plt.savefig(
    SHAP_DIR
    / "Fig_SHAP_top20.png",
    dpi=400,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# REPORT
# ============================================================

print(
    "\nTop 20 predictors by mean |SHAP|:",
    flush=True
)

print(
    imp
    .head(20)
    .round(5)
    .to_string(
        index=False
    ),
    flush=True
)

print(
    "\nNormalised environmental-domain "
    "attribution:",
    flush=True
)

print(
    dom
    .round(4)
    .to_string(
        index=False
    ),
    flush=True
)

print(
    "\nSHAP analysis completed successfully.",
    flush=True
)

print(
    f"Outputs saved to:\n{SHAP_DIR}",
    flush=True
)