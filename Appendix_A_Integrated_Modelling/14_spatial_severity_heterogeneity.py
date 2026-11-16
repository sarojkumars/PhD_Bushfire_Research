# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 18:02:17 2026

@author: sarojkumars
"""

# =============================================================================
# 14_spatial_severity_heterogeneity.py
#
# Appendix A
# Spatial/contextual heterogeneity of RS-derived fire severity
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

from appendix_config import SPATIAL_DATA_FILE as DATA_FILE, HETEROGENEITY_DIR as OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Expected context ordering
LANDFORM_ORDER = [
    "Peak",
    "UpSlp",
    "LwSlp",
    "Valy"
]

VEGETATION_ORDER = [
    "RFVT",
    "EFWL",
    "OFWL",
    "SHMW"
]


# =============================================================================
# 2. READ DATA
# =============================================================================

print("=" * 75)
print("APPENDIX A - SPATIAL/CONTEXTUAL SEVERITY HETEROGENEITY")
print("=" * 75)

df = pd.read_csv(DATA_FILE)

required = [
    "pt_id",
    "RdNBR",
    "SNIC2clsN",
    "LF4Cls",
    "Veg5Cls"
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(
        f"Required columns missing: {missing}"
    )

print(f"\nObservations: {len(df):,}")

print("\nLandform classes:")
print(df["LF4Cls"].value_counts(dropna=False))

print("\nVegetation classes:")
print(df["Veg5Cls"].value_counts(dropna=False))

print("\nSNIC2clsN:")
print(df["SNIC2clsN"].value_counts(dropna=False))


# =============================================================================
# 3. CLEAN NUMERIC RESPONSES
# =============================================================================

df["RdNBR"] = pd.to_numeric(
    df["RdNBR"],
    errors="coerce"
)

df["SNIC2clsN"] = pd.to_numeric(
    df["SNIC2clsN"],
    errors="coerce"
)


# =============================================================================
# 4. DESCRIPTIVE SUMMARY FUNCTION
# =============================================================================

def context_summary(data, context_col):

    temp = data[
        [context_col, "RdNBR", "SNIC2clsN"]
    ].dropna(
        subset=[context_col]
    ).copy()

    records = []

    for context, g in temp.groupby(
        context_col,
        observed=True
    ):

        rdnbr = g["RdNBR"].dropna()

        binary = g["SNIC2clsN"].dropna()

        records.append({

            "context":
                context,

            "n":
                len(g),

            "RdNBR_n":
                len(rdnbr),

            "RdNBR_mean":
                rdnbr.mean(),

            "RdNBR_sd":
                rdnbr.std(),

            "RdNBR_median":
                rdnbr.median(),

            "RdNBR_Q1":
                rdnbr.quantile(0.25),

            "RdNBR_Q3":
                rdnbr.quantile(0.75),

            "RdNBR_IQR":
                (
                    rdnbr.quantile(0.75)
                    -
                    rdnbr.quantile(0.25)
                ),

            "SNIC_n":
                len(binary),

            "high_severity_n":
                int((binary == 1).sum()),

            "high_severity_proportion":
                (
                    (binary == 1).mean()
                    if len(binary) > 0
                    else np.nan
                )
        })

    return pd.DataFrame(records)


land_summary = context_summary(
    df,
    "LF4Cls"
)

veg_summary = context_summary(
    df,
    "Veg5Cls"
)


land_summary.to_csv(
    OUTPUT_DIR /
    "Table_landform_severity_summary.csv",
    index=False
)

veg_summary.to_csv(
    OUTPUT_DIR /
    "Table_vegetation_severity_summary.csv",
    index=False
)


print("\n" + "=" * 75)
print("LANDFORM SUMMARY")
print("=" * 75)
print(land_summary.to_string(index=False))

print("\n" + "=" * 75)
print("VEGETATION SUMMARY")
print("=" * 75)
print(veg_summary.to_string(index=False))


# =============================================================================
# 5. KRUSKAL-WALLIS + EPSILON-SQUARED
# =============================================================================
## =============================================================================

def kruskal_effect(data, context_col):

    temp = data[
        [context_col, "RdNBR"]
    ].dropna()

    groups = []

    labels = []

    for label, g in temp.groupby(
        context_col,
        observed=True
    ):

        values = g["RdNBR"].to_numpy()

        if len(values) > 1:

            groups.append(values)
            labels.append(label)

    H, p = kruskal(*groups)

    n = sum(
        len(x)
        for x in groups
    )

    k = len(groups)

    epsilon_squared = (
        (H - k + 1)
        /
        (n - k)
    )

    epsilon_squared = max(
        0,
        epsilon_squared
    )

    return {
        "context_variable": context_col,
        "n": n,
        "k": k,
        "H": H,
        "p_value": p,
        "epsilon_squared": epsilon_squared
    }


kw_land = kruskal_effect(
    df,
    "LF4Cls"
)

kw_veg = kruskal_effect(
    df,
    "Veg5Cls"
)

kw_results = pd.DataFrame([
    kw_land,
    kw_veg
])

kw_results.to_csv(
    OUTPUT_DIR /
    "Table_Kruskal_Wallis_effect_size.csv",
    index=False
)


print("\n" + "=" * 75)
print("KRUSKAL-WALLIS RESULTS")
print("=" * 75)

print(
    kw_results.to_string(
        index=False
    )
)


# =============================================================================
# 6. PAIRWISE MANN-WHITNEY + RANK-BISERIAL EFFECT SIZE
# =============================================================================
#
# =============================================================================

def holm_adjust(p_values):

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    m = len(p_values)

    order = np.argsort(
        p_values
    )

    sorted_p = p_values[
        order
    ]

    adjusted_sorted = np.empty(
        m,
        dtype=float
    )

    running_max = 0.0

    for i, p in enumerate(sorted_p):

        adjusted = (
            (m - i) * p
        )

        running_max = max(
            running_max,
            adjusted
        )

        adjusted_sorted[i] = min(
            running_max,
            1.0
        )

    adjusted = np.empty(
        m,
        dtype=float
    )

    adjusted[order] = adjusted_sorted

    return adjusted


def pairwise_mannwhitney(
    data,
    context_col
):

    temp = data[
        [context_col, "RdNBR"]
    ].dropna()

    labels = list(
        temp[context_col]
        .drop_duplicates()
    )

    results = []

    for i in range(
        len(labels)
    ):

        for j in range(
            i + 1,
            len(labels)
        ):

            a = labels[i]
            b = labels[j]

            x = temp.loc[
                temp[context_col] == a,
                "RdNBR"
            ].to_numpy()

            y = temp.loc[
                temp[context_col] == b,
                "RdNBR"
            ].to_numpy()

            U, p = mannwhitneyu(
                x,
                y,
                alternative="two-sided"
            )

            r_rb = (
                2 * U /
                (len(x) * len(y))
                - 1
            )

            results.append({

                "context_variable":
                    context_col,

                "group_1":
                    a,

                "group_2":
                    b,

                "n_1":
                    len(x),

                "n_2":
                    len(y),

                "U":
                    U,

                "p_raw":
                    p,

                "rank_biserial":
                    r_rb,

                "abs_rank_biserial":
                    abs(r_rb)
            })

    result_df = pd.DataFrame(
        results
    )

    result_df[
        "p_Holm"
    ] = holm_adjust(
        result_df[
            "p_raw"
        ].values
    )

    return result_df


pair_land = pairwise_mannwhitney(
    df,
    "LF4Cls"
)

pair_veg = pairwise_mannwhitney(
    df,
    "Veg5Cls"
)


pair_land.to_csv(
    OUTPUT_DIR /
    "Table_pairwise_landform_RdNBR.csv",
    index=False
)

pair_veg.to_csv(
    OUTPUT_DIR /
    "Table_pairwise_vegetation_RdNBR.csv",
    index=False
)


# =============================================================================
# 7. HIGH-SEVERITY PROPORTION TEST
# =============================================================================
# =============================================================================

def chi_square_context(
    data,
    context_col
):

    temp = data[
        [context_col, "SNIC2clsN"]
    ].dropna()

    table = pd.crosstab(
        temp[context_col],
        temp["SNIC2clsN"]
    )

    chi2, p, dof, expected = (
        chi2_contingency(table)
    )

    n = table.to_numpy().sum()

    r, k = table.shape

    denom = min(
        r - 1,
        k - 1
    )

    if denom > 0:

        cramers_v = np.sqrt(
            chi2 /
            (n * denom)
        )

    else:

        cramers_v = np.nan

    return {

        "context_variable":
            context_col,

        "n":
            n,

        "chi_square":
            chi2,

        "df":
            dof,

        "p_value":
            p,

        "cramers_V":
            cramers_v
    }


chi_land = chi_square_context(
    df,
    "LF4Cls"
)

chi_veg = chi_square_context(
    df,
    "Veg5Cls"
)


chi_results = pd.DataFrame([
    chi_land,
    chi_veg
])


chi_results.to_csv(
    OUTPUT_DIR /
    "Table_high_severity_context_tests.csv",
    index=False
)


print("\n" + "=" * 75)
print("HIGH-SEVERITY PROPORTION TESTS")
print("=" * 75)

print(
    chi_results.to_string(
        index=False
    )
)


# =============================================================================
# 8. BOXPLOT FUNCTION
# =============================================================================

def make_boxplot(
    data,
    context_col,
    order,
    title,
    filename
):

    available = [
        x for x in order
        if x in set(
            data[context_col]
            .dropna()
            .astype(str)
        )
    ]

    plot_data = []

    labels = []

    for label in available:

        values = (
            data.loc[
                data[context_col].astype(str)
                == label,
                "RdNBR"
            ]
            .dropna()
            .to_numpy()
        )

        if len(values) > 0:

            plot_data.append(
                values
            )

            labels.append(
                label
            )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.boxplot(
        plot_data,
        labels=labels,
        showfliers=False
    )

    ax.set_xlabel(
        "Landscape context"
    )

    ax.set_ylabel(
        "RdNBR"
    )

    ax.set_title(
        title
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


make_boxplot(
    df,
    "LF4Cls",
    LANDFORM_ORDER,
    "EO-derived fire severity across landform contexts",
    "Fig_RdNBR_landform_heterogeneity.png"
)

make_boxplot(
    df,
    "Veg5Cls",
    VEGETATION_ORDER,
    "EO-derived fire severity across vegetation contexts",
    "Fig_RdNBR_vegetation_heterogeneity.png"
)


# =============================================================================
# 9. HIGH-SEVERITY PROPORTION FIGURES
# =============================================================================

def proportion_plot(
    summary_df,
    order,
    title,
    filename
):

    temp = summary_df.copy()

    temp["context"] = (
        temp["context"]
        .astype(str)
    )

    available = [
        x for x in order
        if x in set(
            temp["context"]
        )
    ]

    temp = (
        temp
        .set_index("context")
        .loc[available]
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        temp["context"],
        temp["high_severity_proportion"]
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_xlabel(
        "Landscape context"
    )

    ax.set_ylabel(
        "Proportion classified as high severity"
    )

    ax.set_title(
        title
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


proportion_plot(
    land_summary,
    LANDFORM_ORDER,
    "High-severity proportion across landform contexts",
    "Fig_high_severity_landform.png"
)

proportion_plot(
    veg_summary,
    VEGETATION_ORDER,
    "High-severity proportion across vegetation contexts",
    "Fig_high_severity_vegetation.png"
)


# =============================================================================
# 10. TEXT SUMMARY
# =============================================================================

summary_file = (
    OUTPUT_DIR /
    "Spatial_Severity_Heterogeneity_Summary.txt"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Appendix A - Spatial/contextual heterogeneity "
        "of EO-derived fire severity\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        f"Total observations: {len(df):,}\n\n"
    )

    f.write(
        "LANDFORM SUMMARY\n"
    )

    f.write(
        land_summary.to_string(
            index=False
        )
    )

    f.write(
        "\n\nVEGETATION SUMMARY\n"
    )

    f.write(
        veg_summary.to_string(
            index=False
        )
    )

    f.write(
        "\n\nKRUSKAL-WALLIS\n"
    )

    f.write(
        kw_results.to_string(
            index=False
        )
    )

    f.write(
        "\n\nHIGH-SEVERITY PROPORTION TESTS\n"
    )

    f.write(
        chi_results.to_string(
            index=False
        )
    )

    f.write(
        "\n\nPAIRWISE LANDFORM RdNBR\n"
    )

    f.write(
        pair_land.to_string(
            index=False
        )
    )

    f.write(
        "\n\nPAIRWISE VEGETATION RdNBR\n"
    )

    f.write(
        pair_veg.to_string(
            index=False
        )
    )


print("\n" + "=" * 75)
print("HETEROGENEITY ANALYSIS COMPLETE")
print("=" * 75)

print("\nResults saved to:")
print(OUTPUT_DIR)