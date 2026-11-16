# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 10:48:54 2026

@author: sarojkumars
"""

from pathlib import Path
import pandas as pd
import numpy as np

from appendix_config import *


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

THESIS_DIR = (
    SCRIPT_DIR /
    "appendix_thesis_outputs"
)

TABLE_DIR = (
    THESIS_DIR /
    "tables"
)

ADD_TABLE_DIR = (
    TABLE_DIR /
    "additional"
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

ADD_TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# HELPERS
# =============================================================================

def first_existing(candidates):

    for f in candidates:

        if Path(f).exists():
            return Path(f)

    return None


def save(df, filename, additional=False):

    folder = (
        ADD_TABLE_DIR
        if additional
        else TABLE_DIR
    )

    path = (
        folder /
        filename
    )

    df.to_csv(
        path,
        index=False
    )

    print(
        f"Saved: {path}"
    )


# =============================================================================
# TABLE 6.4
#
# Preserve the original Appendix A logic:
#
# Scenario | Metric | model columns
#
# The corrected analysis uses the validated final models rather than
# reproducing invalid/leaky historical values.
# =============================================================================

def make_table_6_4():

    print(
        "\nPreparing corrected Table 6.4 ..."
    )

    candidates = [
        VAL_DIR /
        "model_performance_by_context.csv",

        VAL_DIR /
        "context_model_performance.csv",

        VAL_DIR /
        "base_and_ensemble_context_metrics.csv",

        VAL_DIR /
        "model_metrics_by_context.csv",

        VAL_DIR /
        "02_model_performance.csv",
    ]

    f = first_existing(
        candidates
    )

    if f is None:

        print(
            "Table 6.4 not generated: "
            "no recognised corrected "
            "model-performance CSV found."
        )

        return

    d = pd.read_csv(f)

    print(
        f"Using:\n{f}"
    )

    # Save source-normalised table first.
    save(
        d,
        "Table_6_4_corrected_model_performance_source.csv"
    )

    # -------------------------------------------------------------------------
    # Try to create thesis-style wide table.
    # -------------------------------------------------------------------------

    scenario_col = next(
        (
            c
            for c in [
                "scenario",
                "context",
                "context_level"
            ]
            if c in d.columns
        ),
        None
    )

    model_col = next(
        (
            c
            for c in [
                "model",
                "classifier"
            ]
            if c in d.columns
        ),
        None
    )

    metric_columns = {
        "Pre": next(
            (
                c
                for c in [
                    "precision",
                    "Precision",
                    "Pre"
                ]
                if c in d.columns
            ),
            None
        ),

        "Rec": next(
            (
                c
                for c in [
                    "recall",
                    "Recall",
                    "Rec"
                ]
                if c in d.columns
            ),
            None
        ),

        "F1": next(
            (
                c
                for c in [
                    "f1",
                    "F1",
                    "f1_score"
                ]
                if c in d.columns
            ),
            None
        ),

        "AUC": next(
            (
                c
                for c in [
                    "auc",
                    "AUC",
                    "roc_auc"
                ]
                if c in d.columns
            ),
            None
        )
    }

    if (
        scenario_col is None
        or
        model_col is None
    ):

        print(
            "Could not automatically pivot "
            "Table 6.4; source CSV retained."
        )

        return

    blocks = []

    for metric_name, metric_col in (
        metric_columns.items()
    ):

        if metric_col is None:
            continue

        p = d.pivot_table(
            index=scenario_col,
            columns=model_col,
            values=metric_col,
            aggfunc="first"
        )

        p.insert(
            0,
            "Metric",
            metric_name
        )

        p = (
            p
            .reset_index()
            .rename(
                columns={
                    scenario_col:
                        "Scenario"
                }
            )
        )

        blocks.append(
            p
        )

    if not blocks:

        print(
            "No recognised Pre/Rec/F1/AUC "
            "columns were found."
        )
        return

    # Interleave metrics by scenario, matching original table logic.
    combined = pd.concat(
        blocks,
        ignore_index=True
    )

    scenario_order = {
        "Overall": 0,
        "Peak": 1,
        "UpSlp": 2,
        "LwSlp": 3,
        "PlnValy": 4,
        "Valy": 4,
        "RFVT": 5,
        "EFWL": 6,
        "OFWL": 7,
        "SHMW": 8
    }

    metric_order = {
        "Pre": 0,
        "Rec": 1,
        "F1": 2,
        "AUC": 3
    }

    combined["_scenario_order"] = (
        combined["Scenario"]
        .map(
            scenario_order
        )
        .fillna(99)
    )

    combined["_metric_order"] = (
        combined["Metric"]
        .map(
            metric_order
        )
        .fillna(99)
    )

    combined = (
        combined
        .sort_values(
            [
                "_scenario_order",
                "_metric_order"
            ]
        )
        .drop(
            columns=[
                "_scenario_order",
                "_metric_order"
            ]
        )
    )

    combined = combined.round(3)

    save(
        combined,
        "Table_6_4_corrected_model_performance.csv"
    )


# =============================================================================
# ADDITIONAL TABLE — ENVIRONMENTAL DOMAIN ABLATION
# =============================================================================

def make_domain_ablation_table():

    print(
        "\nPreparing environmental-domain "
        "ablation table ..."
    )

    candidates = [
        VAL_DIR /
        "environmental_domain_ablation_metrics.csv",

        VAL_DIR /
        "domain_ablation_metrics.csv",

        VAL_DIR /
        "05_environmental_domain_ablation.csv",
    ]

    f = first_existing(
        candidates
    )

    if f is None:

        print(
            "SKIPPED: environmental-domain "
            "ablation table not found."
        )

        return

    d = pd.read_csv(f)

    save(
        d.round(4),
        "Table_additional_environmental_domain_ablation.csv",
        additional=True
    )


# =============================================================================
# ADDITIONAL TABLE — BOOTSTRAP UNCERTAINTY
# =============================================================================

def make_bootstrap_table():

    print(
        "\nPreparing bootstrap uncertainty table ..."
    )

    candidates = (
        list(
            VAL_DIR.glob(
                "*bootstrap*.csv"
            )
        )
        +
        list(
            VAL_DIR.glob(
                "*Bootstrap*.csv"
            )
        )
    )

    if not candidates:

        print(
            "SKIPPED: bootstrap CSV "
            "not found."
        )

        return

    d = pd.read_csv(
        candidates[0]
    )

    save(
        d.round(5),
        "Table_additional_bootstrap_uncertainty.csv",
        additional=True
    )


# =============================================================================
# ADDITIONAL TABLE — CONTEXT ABLATION
# =============================================================================

def make_context_ablation_table():

    print(
        "\nPreparing context-ablation table ..."
    )

    candidates = [
        VAL_DIR /
        "context_ablation_metrics.csv",

        VAL_DIR /
        "attention_context_ablation.csv",

        VAL_DIR /
        "04_context_ablation.csv",
    ]

    f = first_existing(
        candidates
    )

    if f is None:

        print(
            "SKIPPED: context-ablation "
            "summary not found."
        )

        return

    d = pd.read_csv(f)

    save(
        d.round(5),
        "Table_additional_context_ablation.csv",
        additional=True
    )


# =============================================================================
# ADDITIONAL TABLE — EO SEVERITY REPRESENTATION SENSITIVITY
# =============================================================================

def make_eo_sensitivity_table():

    print(
        "\nPreparing EO-response "
        "sensitivity table ..."
    )

    directories = [
        SCRIPT_DIR /
        "appendix_additional_experiments",

        VAL_DIR
    ]

    candidates = []

    for directory in directories:

        if directory.exists():

            candidates += list(
                directory.glob(
                    "*EO*sens*.csv"
                )
            )

            candidates += list(
                directory.glob(
                    "*severity*sens*.csv"
                )
            )

    if not candidates:

        print(
            "SKIPPED: EO sensitivity "
            "summary not found."
        )

        return

    d = pd.read_csv(
        candidates[0]
    )

    save(
        d.round(5),
        "Table_additional_EO_severity_sensitivity.csv",
        additional=True
    )


# =============================================================================
# ADDITIONAL TABLE — SPATIAL AUTOCORRELATION
# =============================================================================

def make_spatial_autocorrelation_table():

    print(
        "\nPreparing spatial "
        "autocorrelation table ..."
    )

    directory = (
        SCRIPT_DIR /
        "appendix_spatial_outputs"
    )

    if not directory.exists():

        print(
            "SKIPPED: spatial output "
            "directory not found."
        )

        return

    candidates = (
        list(
            directory.glob(
                "*moran*.csv"
            )
        )
        +
        list(
            directory.glob(
                "*Moran*.csv"
            )
        )
    )

    if not candidates:

        print(
            "SKIPPED: Moran CSV not found."
        )

        return

    d = pd.read_csv(
        candidates[0]
    )

    save(
        d.round(6),
        "Table_additional_spatial_autocorrelation.csv",
        additional=True
    )


# =============================================================================
# ADDITIONAL TABLE — SPATIAL BLOCK VALIDATION
# =============================================================================

def make_spatial_block_table():

    print(
        "\nPreparing spatial-block "
        "validation table ..."
    )

    directory = (
        SCRIPT_DIR /
        "appendix_spatial_block_validation"
    )

    if not directory.exists():

        print(
            "SKIPPED: spatial-block "
            "directory not found."
        )

        return

    candidates = list(
        directory.glob(
            "*.csv"
        )
    )

    if not candidates:

        print(
            "SKIPPED: spatial-block "
            "CSV not found."
        )

        return

    # Prefer a metrics/summary file.
    preferred = [
        f
        for f in candidates
        if (
            "metric" in f.name.lower()
            or
            "summary" in f.name.lower()
            or
            "validation" in f.name.lower()
        )
    ]

    f = (
        preferred[0]
        if preferred
        else candidates[0]
    )

    d = pd.read_csv(f)

    save(
        d.round(5),
        "Table_additional_spatial_block_validation.csv",
        additional=True
    )


# =============================================================================
# ADDITIONAL TABLE — DOMAIN × CONTEXT
# =============================================================================

def make_domain_context_table():

    print(
        "\nPreparing environmental-domain × "
        "context table ..."
    )

    directory = (
        SCRIPT_DIR /
        "appendix_domain_context"
    )

    if not directory.exists():

        print(
            "SKIPPED: domain-context "
            "directory not found."
        )

        return

    candidates = list(
        directory.glob(
            "*.csv"
        )
    )

    if not candidates:

        print(
            "SKIPPED: domain-context "
            "CSV not found."
        )

        return

    preferred = [
        f
        for f in candidates
        if (
            "metric" in f.name.lower()
            or
            "auc" in f.name.lower()
            or
            "summary" in f.name.lower()
        )
    ]

    f = (
        preferred[0]
        if preferred
        else candidates[0]
    )

    d = pd.read_csv(f)

    save(
        d.round(5),
        "Table_additional_domain_by_context.csv",
        additional=True
    )


# =============================================================================
# INVENTORY
# =============================================================================

# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 78)
    print(
        "APPENDIX A — APPENDIX TABLE OUTPUTS"
    )
    print("=" * 78)

    print(
        f"\nOutput folder:\n"
        f"{TABLE_DIR}"
    )

    make_table_6_4()

    make_domain_ablation_table()
    make_bootstrap_table()
    make_context_ablation_table()
    make_eo_sensitivity_table()
    make_spatial_autocorrelation_table()
    make_spatial_block_table()
    make_domain_context_table()

    print("\n" + "=" * 78)
    print(
        "THESIS TABLE GENERATION COMPLETE"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()