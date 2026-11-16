# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 10:48:21 2026

@author: sarojkumars
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from appendix_config import *


# =============================================================================
# APPENDIX A THESIS OUTPUT DIRECTORY
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

THESIS_DIR = SCRIPT_DIR / "appendix_thesis_outputs"
FIG_DIR = THESIS_DIR / "figures"
ADD_FIG_DIR = FIG_DIR / "additional"

FIG_DIR.mkdir(parents=True, exist_ok=True)
ADD_FIG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# GENERAL THESIS FIGURE SETTINGS
# =============================================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# =============================================================================
# CONTEXT DEFINITIONS
# =============================================================================

CONTEXTS = [
    ("Overall", None, None),
    ("Peak", LANDFORM_COL, "Peak"),
    ("UpSlp", LANDFORM_COL, "UpSlp"),
    ("LwSlp", LANDFORM_COL, "LwSlp"),
    ("PlnValy", LANDFORM_COL, "Valy"),
    ("RFVT", VEGETATION_COL, "RFVT"),
    ("EFWL", VEGETATION_COL, "EFWL"),
    ("OFWL", VEGETATION_COL, "OFWL"),
    ("SHMW", VEGETATION_COL, "SHMW"),
]

DISPLAY_NAME = {
    "avgMaxTempSn": "TMaxSn",
    "avgSurfPresSn": "SLPresSn",
    "TotRainSn": "PrecipSn",
    "TotRainYr": "PrecipYr",
    "avgRHtmaxSn": "RHMaxSn",
    "avgV10Sn": "VWnd",
    "avgU10Sn": "UWnd",
    "avgAbsWndSn": "AbsWnd",
    "avgSwvL1Sn": "SWVL1Sn",
    "SoilMois_0to10": "SMois",

    "DEM": "Elev",
    "SlpPerc": "Slp",
    "AspCosine": "Asp",
    "TWI": "TWI",
    "SolDur": "SolDur",
    "SolDifRad": "SolRad",
    "ConvIndx": "CIndx",
    "ProfCurv": "ProfCurv",
    "PlanCurv": "PlanCurv",
    "LocCurv": "LocCurv",
    "DwnSlpCurv": "DSlpCurv",
    "UpSlpCurv": "USlpCurv",

    "NDMI_mean": "NDMI",
    "LAI": "LAI",
    "LFMC": "LFMC",
    "EVI": "EVI",
    "DFFM_silo": "DFFM",
    "FCOV_BS": "FVCBs",
    "FCOV_PV": "FVCPv",
    "Flamm": "Flam",
    "CHM": "CH",
}


def pretty_feature(x):
    return DISPLAY_NAME.get(x, x)


# =============================================================================
# HELPERS
# =============================================================================

def save_figure(fig, path):
    fig.savefig(
        path,
        dpi=400,
        bbox_inches="tight"
    )
    plt.close(fig)

    print(f"Saved: {path}")


def load_main_data():
    if not Path(DATA_FILE).exists():
        raise FileNotFoundError(
            f"Main data not found:\n{DATA_FILE}"
        )

    return pd.read_csv(DATA_FILE)


def get_context_subset(df, context_name, column, level):

    if column is None:
        return df.copy()

    if context_name == "PlnValy":
        # Current modelling data may use Valy while thesis uses PlnValy.
        possible = ["Valy", "PlnValy"]

        return df[
            df[column].astype(str).isin(possible)
        ].copy()

    return df[
        df[column].astype(str) == str(level)
    ].copy()


# =============================================================================
# FIGURE 6.5
# ORIGINAL CHAPTER STRUCTURE:
# Overall + four landforms + four vegetation groups
# =============================================================================

def make_figure_6_5():

    print("\nGenerating Figure 6.5 ...")

    df = load_main_data()

    # Appendix A describes this figure using RdNBR.
    if "RdNBR" not in df.columns:
        print(
            "SKIPPED Figure 6.5: RdNBR not found."
        )
        return

    labels = []
    values = []

    for name, col, level in CONTEXTS:

        sub = get_context_subset(
            df,
            name,
            col,
            level
        )

        x = pd.to_numeric(
            sub["RdNBR"],
            errors="coerce"
        ).dropna()

        labels.append(name)
        values.append(x.values)

    fig, ax = plt.subplots(
        figsize=(11.5, 4.8)
    )

    bp = ax.boxplot(
        values,
        labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.58,
        medianprops={
            "linewidth": 1.1
        }
    )

    # Keep restrained styling similar to thesis.
    for patch in bp["boxes"]:
        patch.set_alpha(0.65)

    # Circumscribed/open red mean marker as in original Fig. 6.5.
    means = [
        np.nanmean(v)
        if len(v) > 0
        else np.nan
        for v in values
    ]

    ax.scatter(
        np.arange(
            1,
            len(means) + 1
        ),
        means,
        marker="o",
        facecolors="none",
        edgecolors="red",
        s=38,
        linewidths=1.1,
        zorder=5,
        label="Mean"
    )

    ax.set_ylabel("Fire severity (RdNBR)")
    ax.set_xlabel("Scenario")

    ax.tick_params(
        axis="x",
        rotation=0
    )

    ax.grid(
        axis="y",
        alpha=0.20
    )

    ax.legend(
        frameon=False,
        loc="upper right"
    )

    fig.tight_layout()

    save_figure(
        fig,
        FIG_DIR /
        "Fig_6_5_severity_distribution.png"
    )


# =============================================================================
# LOAD SHAP OUTPUTS
# =============================================================================

def load_shap_outputs():

    shap_values_file = (
        SHAP_DIR /
        "shap_values.npy"
    )

    shap_sample_file = (
        SHAP_DIR /
        "shap_sample.csv"
    )

    if not shap_values_file.exists():
        return None, None

    if not shap_sample_file.exists():
        return None, None

    sv = np.load(
        shap_values_file
    )

    sample = pd.read_csv(
        shap_sample_file
    )

    if len(sample) != sv.shape[0]:
        raise RuntimeError(
            "SHAP sample and SHAP-value row counts "
            "do not match."
        )

    if sv.shape[1] != len(ALL_PREDICTORS):
        raise RuntimeError(
            "SHAP predictor count does not match "
            "ALL_PREDICTORS."
        )

    return sample, sv


# =============================================================================
# CUSTOM SHAP BEESWARM
#
# This recreates the visual logic of original Figure 6.6:
# SHAP value on x-axis, predictors on y-axis,
# point colour represents feature value.
#
# It does NOT call shap.TreeExplainer, so the XGBoost 3.2 /
# SHAP 0.49 parser issue does not affect plotting.
# =============================================================================

def draw_beeswarm(
    ax,
    X,
    shap_values,
    title,
    max_features=10,
    point_size=5
):

    if len(X) == 0:
        ax.axis("off")
        return

    mean_abs = (
        np.abs(shap_values)
        .mean(axis=0)
    )

    order = np.argsort(
        mean_abs
    )[::-1][:max_features]

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    for row_index, feature_index in enumerate(
        order
    ):

        feature = ALL_PREDICTORS[
            feature_index
        ]

        shap_x = shap_values[
            :,
            feature_index
        ]

        feature_x = pd.to_numeric(
            X[feature],
            errors="coerce"
        ).to_numpy()

        finite = (
            np.isfinite(shap_x)
            &
            np.isfinite(feature_x)
        )

        shap_x = shap_x[finite]
        feature_x = feature_x[finite]

        if len(shap_x) == 0:
            continue

        # Robust feature-value scaling for colour.
        lo = np.nanpercentile(
            feature_x,
            5
        )

        hi = np.nanpercentile(
            feature_x,
            95
        )

        if not np.isfinite(lo):
            lo = np.nanmin(feature_x)

        if not np.isfinite(hi):
            hi = np.nanmax(feature_x)

        if hi <= lo:
            colour_value = np.full(
                len(feature_x),
                0.5
            )
        else:
            colour_value = (
                feature_x - lo
            ) / (
                hi - lo
            )

            colour_value = np.clip(
                colour_value,
                0,
                1
            )

        jitter = rng.normal(
            loc=0.0,
            scale=0.085,
            size=len(shap_x)
        )

        y = (
            np.full(
                len(shap_x),
                row_index
            )
            +
            jitter
        )

        ax.scatter(
            shap_x,
            y,
            c=colour_value,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=point_size,
            alpha=0.65,
            linewidths=0,
            rasterized=True
        )

    labels = [
        pretty_feature(
            ALL_PREDICTORS[i]
        )
        for i in order
    ]

    ax.set_yticks(
        np.arange(
            len(order)
        )
    )

    ax.set_yticklabels(
        labels
    )

    ax.invert_yaxis()

    ax.axvline(
        0,
        linewidth=0.6,
        alpha=0.5
    )

    ax.set_title(
        title,
        fontweight="bold"
    )

    ax.set_xlabel(
        "SHAP value (impact on model output)"
    )


# =============================================================================
# FIGURE 6.6
# ORIGINAL STRUCTURE:
# large Overall panel + 8 smaller scenario panels
# =============================================================================

def make_figure_6_6():

    print("\nGenerating Figure 6.6 ...")

    sample, sv = load_shap_outputs()

    if sample is None:
        print(
            "SKIPPED Figure 6.6: "
            "SHAP outputs not available."
        )
        return

    # Original figure:
    # Overall occupies left side;
    # 8 context panels arranged on right.
    fig = plt.figure(
        figsize=(14, 10)
    )

    gs = fig.add_gridspec(
        nrows=4,
        ncols=3,
        width_ratios=[
            1.25,
            1,
            1
        ],
        hspace=0.52,
        wspace=0.60
    )

    ax_overall = fig.add_subplot(
        gs[:, 0]
    )

    draw_beeswarm(
        ax_overall,
        sample,
        sv,
        "Overall",
        max_features=len(
            ALL_PREDICTORS
        ),
        point_size=5
    )

    scenario_axes = []

    for r in range(4):
        for c in [
            1,
            2
        ]:
            scenario_axes.append(
                fig.add_subplot(
                    gs[r, c]
                )
            )

    context_defs = CONTEXTS[1:]

    for ax, (
        name,
        col,
        level
    ) in zip(
        scenario_axes,
        context_defs
    ):

        if name == "PlnValy":
            mask = (
                sample[col]
                .astype(str)
                .isin(
                    [
                        "Valy",
                        "PlnValy"
                    ]
                )
                .to_numpy()
            )
        else:
            mask = (
                sample[col]
                .astype(str)
                ==
                str(level)
            ).to_numpy()

        n = int(mask.sum())

        if n == 0:
            ax.axis("off")
            ax.set_title(name)
            continue

        draw_beeswarm(
            ax,
            sample.loc[
                mask,
                ALL_PREDICTORS
            ],
            sv[mask],
            f"{name} (n={n:,})",
            max_features=10,
            point_size=4
        )

    # Shared feature-value colour bar.
    sm = ScalarMappable(
        norm=Normalize(
            vmin=0,
            vmax=1
        ),
        cmap="coolwarm"
    )

    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=fig.axes,
        fraction=0.015,
        pad=0.015
    )

    cbar.set_ticks(
        [
            0,
            1
        ]
    )

    cbar.set_ticklabels(
        [
            "Low",
            "High"
        ]
    )

    cbar.set_label(
        "Feature value"
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.91,
        bottom=0.07,
        top=0.96
    )

    save_figure(
        fig,
        FIG_DIR /
        "Fig_6_6_SHAP_scenarios.png"
    )


# =============================================================================
# FIGURE 6.7
# a) domain-level attribution
# b) scenario-specific individual predictor attribution
# =============================================================================

def make_figure_6_7():

    print("\nGenerating Figure 6.7 ...")

    sample, sv = load_shap_outputs()

    if sample is None:
        print(
            "SKIPPED Figure 6.7: "
            "SHAP outputs not available."
        )
        return

    domain_lookup = {
        **{
            f: "Meteorological"
            for f in METEOROLOGY
        },
        **{
            f: "Topographical"
            for f in TOPOGRAPHY
        },
        **{
            f: "Fuel/vegetation"
            for f in FUEL_VEGETATION
        }
    }

    domain_order = [
        "Fuel/vegetation",
        "Meteorological",
        "Topographical"
    ]

    domain_rows = []
    feature_rows = []

    for name, col, level in CONTEXTS:

        if col is None:
            mask = np.ones(
                len(sample),
                dtype=bool
            )

        elif name == "PlnValy":
            mask = (
                sample[col]
                .astype(str)
                .isin(
                    [
                        "Valy",
                        "PlnValy"
                    ]
                )
                .to_numpy()
            )

        else:
            mask = (
                sample[col]
                .astype(str)
                ==
                str(level)
            ).to_numpy()

        if mask.sum() == 0:
            continue

        a = (
            np.abs(
                sv[mask]
            )
            .mean(axis=0)
        )

        total = a.sum()

        if total <= 0:
            continue

        rel = a / total

        tmp = pd.DataFrame({
            "feature":
                ALL_PREDICTORS,

            "importance":
                rel
        })

        tmp["domain"] = (
            tmp["feature"]
            .map(domain_lookup)
        )

        for _, row in tmp.iterrows():

            feature_rows.append({
                "scenario":
                    name,

                "feature":
                    row["feature"],

                "importance":
                    row["importance"],

                "domain":
                    row["domain"]
            })

        d = (
            tmp
            .groupby(
                "domain"
            )["importance"]
            .sum()
        )

        for domain in domain_order:

            domain_rows.append({
                "scenario":
                    name,

                "domain":
                    domain,

                "importance":
                    d.get(
                        domain,
                        0.0
                    )
            })

    domain_df = pd.DataFrame(
        domain_rows
    )

    feature_df = pd.DataFrame(
        feature_rows
    )

    # Save plotting data for complete reproducibility.
    domain_df.to_csv(
        FIG_DIR /
        "Fig_6_7a_plot_data.csv",
        index=False
    )

    feature_df.to_csv(
        FIG_DIR /
        "Fig_6_7b_plot_data.csv",
        index=False
    )

    fig = plt.figure(
        figsize=(13, 11)
    )

    gs = fig.add_gridspec(
        nrows=6,
        ncols=2,
        height_ratios=[
            1.35,
            1,
            1,
            1,
            1,
            1
        ],
        hspace=0.75,
        wspace=0.25
    )

    # -------------------------------------------------------------------------
    # PANEL A
    # -------------------------------------------------------------------------

    ax_a = fig.add_subplot(
        gs[0, :]
    )

    scenarios = [
        x[0]
        for x in CONTEXTS
    ]

    x = np.arange(
        len(scenarios)
    )

    width = 0.24

    for j, domain in enumerate(
        domain_order
    ):

        vals = []

        for scenario in scenarios:

            q = domain_df[
                (
                    domain_df["scenario"]
                    ==
                    scenario
                )
                &
                (
                    domain_df["domain"]
                    ==
                    domain
                )
            ]

            vals.append(
                q["importance"].iloc[0]
                if len(q)
                else np.nan
            )

        ax_a.bar(
            x
            +
            (
                j - 1
            )
            *
            width,
            vals,
            width=width,
            label=domain
        )

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(
        scenarios
    )

    ax_a.set_ylabel(
        "Relative importance"
    )

    ax_a.set_title(
        "a)",
        loc="left",
        fontweight="bold"
    )

    ax_a.legend(
        frameon=False,
        ncol=3,
        loc="upper right"
    )

    ax_a.grid(
        axis="y",
        alpha=0.18
    )

    # -------------------------------------------------------------------------
    # PANEL B
    # -------------------------------------------------------------------------

    axes = []

    for r in range(
        1,
        6
    ):
        for c in range(2):
            axes.append(
                fig.add_subplot(
                    gs[r, c]
                )
            )

    # Only nine contexts; last axis is unused.
    for ax, scenario in zip(
        axes,
        scenarios
    ):

        q = (
            feature_df[
                feature_df[
                    "scenario"
                ]
                ==
                scenario
            ]
            .sort_values(
                "importance",
                ascending=False
            )
            .head(15)
            .copy()
        )

        if q.empty:
            ax.axis("off")
            continue

        # Preserve environmental-domain distinction.
        # Matplotlib default colour cycle is used deliberately.
        domain_codes = {
            d: i
            for i, d in enumerate(
                domain_order
            )
        }

        colours = [
            f"C{domain_codes[d]}"
            for d in q["domain"]
        ]

        ax.bar(
            np.arange(
                len(q)
            ),
            q["importance"],
            color=colours
        )

        ax.set_xticks(
            np.arange(
                len(q)
            )
        )

        ax.set_xticklabels(
            [
                pretty_feature(x)
                for x in q["feature"]
            ],
            rotation=55,
            ha="right"
        )

        ax.set_title(
            scenario,
            loc="right",
            fontweight="bold"
        )

        ax.set_ylabel(
            "Relative importance"
        )

        ax.grid(
            axis="y",
            alpha=0.15
        )

    if len(axes) > len(scenarios):
        for ax in axes[
            len(scenarios):
        ]:
            ax.axis("off")

    # "b)" marker positioned before first individual panel.
    axes[0].text(
        -0.12,
        1.18,
        "b)",
        transform=axes[0].transAxes,
        fontweight="bold",
        fontsize=10
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.98,
        bottom=0.06,
        top=0.97
    )

    save_figure(
        fig,
        FIG_DIR /
        "Fig_6_7_SHAP_importance.png"
    )


# =============================================================================
# FIGURE 6.8
#
# The original GAM figure is highly specialised and should NOT be replaced by
# an invented simplified figure.
#
# This function looks for the existing GAM composite generated by the GAM
# analysis. If found, it copies it into the thesis output folder.
#
# If the GAM script produces component plots rather than a composite,
# leave the original figure unchanged until the GAM output is inspected.
# =============================================================================

def make_figure_6_8():

    print("\nPreparing Figure 6.8 ...")

    candidates = [
        GAM_DIR / "Fig_GAM.png",
        GAM_DIR / "Fig_GAM_final.png",
        GAM_DIR / "GAM_combined.png",
        GAM_DIR / "GAM_outputs.png",
        GAM_DIR / "Fig_6_8_GAM.png",
    ]

    existing = [
        x
        for x in candidates
        if x.exists()
    ]

    if not existing:

        print(
            "Figure 6.8 not regenerated automatically. "
            "No verified combined GAM figure was found. "
            "The existing thesis Figure 6.8 should be "
            "retained until GAM outputs are inspected."
        )

        return

    import shutil

    source = existing[0]

    target = (
        FIG_DIR /
        "Fig_6_8_GAM.png"
    )

    shutil.copy2(
        source,
        target
    )

    print(
        f"Copied existing GAM figure:\n"
        f"{source}\n->\n{target}"
    )


# =============================================================================
# ADDITIONAL FIGURE: ENVIRONMENTAL-DOMAIN ABLATION
# =============================================================================

def make_additional_domain_figure():

    print(
        "\nGenerating additional "
        "environmental-domain figure ..."
    )

    candidates = [
        VAL_DIR /
        "environmental_domain_ablation_metrics.csv",

        VAL_DIR /
        "domain_ablation_metrics.csv",

        VAL_DIR /
        "05_environmental_domain_ablation.csv",
    ]

    f = next(
        (
            x
            for x in candidates
            if x.exists()
        ),
        None
    )

    if f is None:
        print(
            "SKIPPED: domain-ablation "
            "summary CSV not found."
        )
        return

    d = pd.read_csv(f)

    # Flexible column detection.
    domain_col = next(
        (
            c
            for c in [
                "domain_set",
                "feature_set",
                "domain",
                "model"
            ]
            if c in d.columns
        ),
        None
    )

    auc_col = next(
        (
            c
            for c in [
                "AUC",
                "auc",
                "roc_auc"
            ]
            if c in d.columns
        ),
        None
    )

    if (
        domain_col is None
        or
        auc_col is None
    ):
        print(
            "SKIPPED: could not identify "
            "domain/AUC columns in "
            f"{f.name}"
        )
        return

    fig, ax = plt.subplots(
        figsize=(8.5, 4.8)
    )

    ax.bar(
        d[domain_col].astype(str),
        d[auc_col]
    )

    ax.set_ylabel("AUC")
    ax.set_xlabel(
        "Environmental information"
    )

    ax.tick_params(
        axis="x",
        rotation=35
    )

    ax.grid(
        axis="y",
        alpha=0.18
    )

    fig.tight_layout()

    save_figure(
        fig,
        ADD_FIG_DIR /
        "Additional_environmental_domain_ablation.png"
    )


# =============================================================================
# ADDITIONAL FIGURE: SPATIAL AUTOCORRELATION
# =============================================================================

def make_additional_spatial_autocorrelation():

    print(
        "\nGenerating additional spatial "
        "autocorrelation figure ..."
    )

    spatial_dir = (
        SCRIPT_DIR /
        "appendix_spatial_outputs"
    )

    candidates = list(
        spatial_dir.glob(
            "*moran*.csv"
        )
    ) + list(
        spatial_dir.glob(
            "*Moran*.csv"
        )
    )

    if not candidates:
        print(
            "SKIPPED: Moran's I output "
            "CSV not found."
        )
        return

    f = candidates[0]

    d = pd.read_csv(f)

    distance_col = next(
        (
            c
            for c in d.columns
            if "dist" in c.lower()
        ),
        None
    )

    moran_col = next(
        (
            c
            for c in d.columns
            if (
                "moran" in c.lower()
                and
                "expected" not in c.lower()
            )
        ),
        None
    )

    variable_col = next(
        (
            c
            for c in [
                "variable",
                "response",
                "target"
            ]
            if c in d.columns
        ),
        None
    )

    if (
        distance_col is None
        or
        moran_col is None
    ):
        print(
            "SKIPPED: Moran output column "
            "structure not recognised."
        )
        return

    fig, ax = plt.subplots(
        figsize=(8.5, 4.8)
    )

    if variable_col is not None:

        for name, g in d.groupby(
            variable_col
        ):

            ax.plot(
                g[distance_col],
                g[moran_col],
                marker="o",
                label=str(name)
            )

        ax.legend(
            frameon=False
        )

    else:

        ax.plot(
            d[distance_col],
            d[moran_col],
            marker="o"
        )

    ax.axhline(
        0,
        linewidth=0.8,
        alpha=0.5
    )

    ax.set_xlabel(
        "Distance class"
    )

    ax.set_ylabel(
        "Moran's I"
    )

    ax.grid(
        alpha=0.18
    )

    fig.tight_layout()

    save_figure(
        fig,
        ADD_FIG_DIR /
        "Additional_spatial_autocorrelation.png"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 78)
    print(
        "APPENDIX A — ORIGINAL-STYLE "
        "THESIS FIGURES"
    )
    print("=" * 78)

    print(
        f"\nOutput folder:\n{FIG_DIR}"
    )

    make_figure_6_5()
    make_figure_6_6()
    make_figure_6_7()
    make_figure_6_8()

    make_additional_domain_figure()
    make_additional_spatial_autocorrelation()

    print("\n" + "=" * 78)
    print(
        "THESIS FIGURE GENERATION COMPLETE"
    )
    print("=" * 78)

    print(
        "\nNOTE:"
        "\nFigures 6.1–6.4 remain the existing "
        "Appendix A methodological figures."
        "\nFigure 6.8 is not reconstructed unless "
        "a verified GAM composite is available."
    )


if __name__ == "__main__":
    main()