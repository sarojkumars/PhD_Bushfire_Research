# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 18:02:51 2026

@author: sarojkumars
"""

# =============================================================================
# 15_spatial_block_validation.py
#
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
    accuracy_score,
    precision_score,
    recall_score
)

from xgboost import XGBClassifier


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

from appendix_config import (
    SPATIAL_DATA_FILE as DATA_FILE,
    BLOCK_DIR as OUTPUT_DIR,
    XGB_MODEL_FILE as EXISTING_XGB_MODEL,
    ALL_PREDICTORS as FEATURES,
    TARGET,
    RANDOM_STATE,
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# Spatial sensitivity scales
BLOCK_SIZES = [
    5000,
    10000
]


# Number of spatial folds
N_SPLITS = 5


RANDOM_STATE = 42


# Predictor list, target, paths and random state are imported from appendix_config.py.

# =============================================================================
# 3. READ DATA
# =============================================================================

print("=" * 78)
print("APPENDIX A - SPATIAL-BLOCK VALIDATION")
print("=" * 78)

df = pd.read_csv(
    DATA_FILE
)


required = (
    FEATURES
    +
    [
        TARGET,
        "x_cood",
        "y_cood",
        "pt_id"
    ]
)


missing = [
    c for c in required
    if c not in df.columns
]


if missing:

    raise ValueError(
        f"Required columns missing:\n{missing}"
    )


print(
    f"\nObservations: {len(df):,}"
)


# =============================================================================
# 4. PREPARE X / y
# =============================================================================

X = (
    df[FEATURES]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
)

y = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)


valid_target = y.notna()


if not valid_target.all():

    print(
        f"Removing {(~valid_target).sum():,} "
        "rows with missing target."
    )

    df = (
        df.loc[valid_target]
        .reset_index(drop=True)
    )

    X = (
        X.loc[valid_target]
        .reset_index(drop=True)
    )

    y = (
        y.loc[valid_target]
        .reset_index(drop=True)
    )


y = y.astype(int)


print(
    "\nTarget distribution:"
)

print(
    y.value_counts(
        normalize=False
    ).sort_index()
)


# =============================================================================
# 5. LOAD EXISTING XGB PARAMETERS IF AVAILABLE
# =============================================================================

def extract_xgb_params(model):

    # Pipeline case
    if hasattr(
        model,
        "named_steps"
    ):

        for step_name, step in (
            model.named_steps.items()
        ):

            if isinstance(
                step,
                XGBClassifier
            ):

                return step.get_params()

    # Direct XGBClassifier
    if isinstance(
        model,
        XGBClassifier
    ):

        return model.get_params()

    return None


if EXISTING_XGB_MODEL.exists():

    print(
        "\nLoading existing XGB model:"
    )

    print(
        EXISTING_XGB_MODEL
    )

    loaded_model = joblib.load(
        EXISTING_XGB_MODEL
    )

    xgb_params = extract_xgb_params(
        loaded_model
    )

else:

    xgb_params = None


# -----------------------------------------------------------------------------
# Fallback only if existing model cannot be loaded.
#
# -----------------------------------------------------------------------------

if xgb_params is None:

    raise FileNotFoundError(
        "Final XGB parameters could not be loaded. Run Script 02 first and "
        f"confirm the model exists at: {EXISTING_XGB_MODEL}"
    )



safe_params = {
    k: v
    for k, v in xgb_params.items()
    if k not in [
        "missing"
    ]
}


safe_params[
    "random_state"
] = RANDOM_STATE

safe_params[
    "n_jobs"
] = -1


print(
    "\nXGB parameters used:"
)

for key in sorted(
    safe_params
):

    print(
        f"{key}: "
        f"{safe_params[key]}"
    )


# =============================================================================
# 6. CREATE SPATIAL BLOCKS
# =============================================================================
#
# Blocks are anchored to the minimum x/y coordinate.
#
# A unique block ID is then created.
# =============================================================================

def create_blocks(
    data,
    block_size
):

    x = data[
        "x_cood"
    ].to_numpy(
        dtype=float
    )

    ycoord = data[
        "y_cood"
    ].to_numpy(
        dtype=float
    )

    xmin = np.min(x)
    ymin = np.min(ycoord)

    xb = np.floor(
        (x - xmin)
        /
        block_size
    ).astype(int)

    yb = np.floor(
        (ycoord - ymin)
        /
        block_size
    ).astype(int)

    block_string = (
        pd.Series(xb).astype(str)
        +
        "_"
        +
        pd.Series(yb).astype(str)
    )

    block_codes, uniques = (
        pd.factorize(
            block_string
        )
    )

    return (
        block_codes,
        xb,
        yb
    )


# =============================================================================
# 7. METRIC FUNCTION
# =============================================================================

def calculate_metrics(
    y_true,
    probability
):

    prediction = (
        probability >= 0.5
    ).astype(int)

    return {

        "AUC":
            roc_auc_score(
                y_true,
                probability
            ),

        "accuracy":
            accuracy_score(
                y_true,
                prediction
            ),

        "precision_class1":
            precision_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "recall_class1":
            recall_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "F1_class1":
            f1_score(
                y_true,
                prediction,
                zero_division=0
            ),

        "macro_F1":
            f1_score(
                y_true,
                prediction,
                average="macro",
                zero_division=0
            ),

        "MCC":
            matthews_corrcoef(
                y_true,
                prediction
            )
    }


# =============================================================================
# 8. SPATIAL BLOCK CROSS-VALIDATION
# =============================================================================

all_fold_results = []
all_predictions = []
block_assignment_records = []


for block_size in BLOCK_SIZES:

    print(
        "\n" + "=" * 78
    )

    print(
        f"SPATIAL BLOCK SIZE: "
        f"{block_size:,} coordinate units"
    )

    print(
        "=" * 78
    )


    (
        groups,
        xb,
        yb
    ) = create_blocks(
        df,
        block_size
    )


    n_blocks = len(
        np.unique(groups)
    )


    print(
        f"Number of occupied blocks: "
        f"{n_blocks:,}"
    )


    if n_blocks < N_SPLITS:

        raise ValueError(
            f"Only {n_blocks} blocks "
            f"for {N_SPLITS} folds."
        )


    block_assignment = pd.DataFrame({

        "pt_id":
            df["pt_id"].values,

        "block_size":
            block_size,

        "x_block":
            xb,

        "y_block":
            yb,

        "block_id":
            groups
    })


    block_assignment_records.append(
        block_assignment
    )


    cv = GroupKFold(
        n_splits=N_SPLITS
    )


    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        cv.split(
            X,
            y,
            groups=groups
        ),
        start=1
    ):

        print(
            f"\nBlock {block_size:,} "
            f"- Fold {fold}/{N_SPLITS}"
        )

        print(
            f"Train: {len(train_idx):,}"
        )

        print(
            f"Test:  {len(test_idx):,}"
        )


        y_train = y.iloc[
            train_idx
        ]

        y_test = y.iloc[
            test_idx
        ]


        print(
            "Train class-1 proportion:",
            round(
                y_train.mean(),
                4
            )
        )

        print(
            "Test class-1 proportion:",
            round(
                y_test.mean(),
                4
            )
        )


        # ---------------------------------------------------------
        # Pipeline
        # ---------------------------------------------------------

        model = Pipeline([

            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "model",
                XGBClassifier(
                    **safe_params
                )
            )
        ])


        model.fit(
            X.iloc[
                train_idx
            ],
            y_train
        )


        probability = (
            model.predict_proba(
                X.iloc[
                    test_idx
                ]
            )[:, 1]
        )


        metrics = calculate_metrics(
            y_test,
            probability
        )


        result = {

            "block_size":
                block_size,

            "fold":
                fold,

            "n_train":
                len(train_idx),

            "n_test":
                len(test_idx),

            "n_blocks":
                n_blocks,

            "train_prevalence":
                y_train.mean(),

            "test_prevalence":
                y_test.mean(),

            **metrics
        }


        all_fold_results.append(
            result
        )


        prediction_df = pd.DataFrame({

            "pt_id":
                df.iloc[
                    test_idx
                ]["pt_id"].values,

            "block_size":
                block_size,

            "fold":
                fold,

            "y_true":
                y_test.values,

            "probability":
                probability
        })


        all_predictions.append(
            prediction_df
        )


        print(
            f"AUC={metrics['AUC']:.4f}, "
            f"macro-F1={metrics['macro_F1']:.4f}, "
            f"MCC={metrics['MCC']:.4f}"
        )


# =============================================================================
# 9. SAVE FOLD RESULTS
# =============================================================================

fold_results = pd.DataFrame(
    all_fold_results
)


fold_results.to_csv(
    OUTPUT_DIR /
    "spatial_block_validation_folds.csv",
    index=False
)


predictions = pd.concat(
    all_predictions,
    ignore_index=True
)


predictions.to_csv(
    OUTPUT_DIR /
    "spatial_block_validation_predictions.csv",
    index=False
)


block_assignments = pd.concat(
    block_assignment_records,
    ignore_index=True
)


block_assignments.to_csv(
    OUTPUT_DIR /
    "spatial_block_assignments.csv",
    index=False
)


# =============================================================================
# 10. SUMMARY ACROSS FOLDS
# =============================================================================

metric_columns = [
    "AUC",
    "accuracy",
    "precision_class1",
    "recall_class1",
    "F1_class1",
    "macro_F1",
    "MCC"
]


summary_records = []


for block_size in BLOCK_SIZES:

    subset = fold_results[
        fold_results[
            "block_size"
        ]
        ==
        block_size
    ]


    for metric in metric_columns:

        values = subset[
            metric
        ].to_numpy()

        summary_records.append({

            "block_size":
                block_size,

            "metric":
                metric,

            "mean":
                np.mean(values),

            "sd":
                np.std(
                    values,
                    ddof=1
                ),

            "minimum":
                np.min(values),

            "maximum":
                np.max(values)
        })


summary_df = pd.DataFrame(
    summary_records
)


summary_df.to_csv(
    OUTPUT_DIR /
    "Table_spatial_block_validation_summary.csv",
    index=False
)


# =============================================================================
# 11. POOLED OUT-OF-FOLD PERFORMANCE
# =============================================================================
## Every observation is predicted once for each block-size analysis.
# This gives a pooled out-of-fold metric for each spatial scale.
# =============================================================================

pooled_records = []


for block_size in BLOCK_SIZES:

    temp = predictions[
        predictions[
            "block_size"
        ]
        ==
        block_size
    ]


    metrics = calculate_metrics(
        temp["y_true"].to_numpy(),
        temp["probability"].to_numpy()
    )


    pooled_records.append({

        "block_size":
            block_size,

        "n":
            len(temp),

        **metrics
    })


pooled_df = pd.DataFrame(
    pooled_records
)


pooled_df.to_csv(
    OUTPUT_DIR /
    "Table_spatial_block_validation_pooled.csv",
    index=False
)


print("\n" + "=" * 78)
print("POOLED SPATIAL VALIDATION")
print("=" * 78)

print(
    pooled_df.to_string(
        index=False
    )
)


# =============================================================================
# 12. COMPARISON FIGURE
# =============================================================================

plot_metrics = [
    "AUC",
    "macro_F1",
    "MCC"
]


for metric in plot_metrics:

    temp = summary_df[
        summary_df[
            "metric"
        ]
        ==
        metric
    ].copy()


    fig, ax = plt.subplots(
        figsize=(7, 5)
    )


    ax.errorbar(
        temp[
            "block_size"
        ].astype(str),
        temp[
            "mean"
        ],
        yerr=temp[
            "sd"
        ],
        marker="o",
        capsize=4
    )


    ax.set_xlabel(
        "Spatial block size"
    )


    ax.set_ylabel(
        metric
    )


    ax.set_title(
        f"XGB spatial-block validation: {metric}"
    )


    fig.tight_layout()


    fig.savefig(
        OUTPUT_DIR /
        f"Fig_spatial_block_{metric}.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close(fig)


# =============================================================================
# 13. TEXT SUMMARY
# =============================================================================

summary_file = (
    OUTPUT_DIR /
    "Spatial_Block_Validation_Summary.txt"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Appendix A - XGB spatial-block validation sensitivity\n"
    )

    f.write(
        "=" * 78 + "\n\n"
    )

    f.write(
        f"Observations: {len(df):,}\n"
    )

    f.write(
        f"Predictors: {len(FEATURES)}\n"
    )

    f.write(
        f"Spatial folds: {N_SPLITS}\n\n"
    )


    f.write(
        "POOLED OUT-OF-FOLD PERFORMANCE\n"
    )

    f.write(
        pooled_df.to_string(
            index=False
        )
    )


    f.write(
        "\n\nFOLD-LEVEL PERFORMANCE\n"
    )

    f.write(
        fold_results.to_string(
            index=False
        )
    )


    f.write(
        "\n\nMEAN +/- SD ACROSS FOLDS\n"
    )

    f.write(
        summary_df.to_string(
            index=False
        )
    )


print("\n" + "=" * 78)
print("SPATIAL-BLOCK VALIDATION COMPLETE")
print("=" * 78)

print("\nResults saved to:")
print(OUTPUT_DIR)