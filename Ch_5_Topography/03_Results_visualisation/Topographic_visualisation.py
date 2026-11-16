
"""Visualise RdNBR distributions across topographic classes."""

import os
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    path = Path(
        os.environ.get(
            "TOPO_VIS_CSV",
            "topography_visualisation_input.csv"
        )
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Set TOPO_VIS_CSV to the prepared plotting table: {path}"
        )

    data = pd.read_csv(path)

    required = {"Lform", "AspClass", "ProfCurvCls1", "RdNBR"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing plotting columns: {sorted(missing)}"
        )

    data = data.dropna(subset=list(required)).copy()
    data["RdNBR"] = pd.to_numeric(data["RdNBR"], errors="raise")

    sns.set_theme(style="ticks", context="paper")

    # 1. RdNBR distribution across landform classes.
    # Class codes and labels follow the prepared GIS dataset.
    landform_order = [5, 6, 7, 2, 1, 4, 9, 8, 10, 3]
    landform_names = [
        "PL", "OS", "US", "MSD", "CN",
        "USV", "MSR", "LR", "HR", "ULD"
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.violinplot(
        data=data,
        x="Lform",
        y="RdNBR",
        order=landform_order,
        inner="box",
        ax=ax
    )

    ax.set_xticks(
        range(len(landform_names)),
        labels=landform_names,
        rotation=65
    )
    ax.set(
        xlabel="Landform types",
        ylabel="Burn severity (RdNBR)"
    )
    fig.tight_layout()

    # 2. RdNBR distribution by aspect class.
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.boxplot(
        data=data,
        x="AspClass",
        y="RdNBR",
        order=["N", "S"],
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "black"
        },
        ax=ax
    )

    ax.set_xticks(
        [0, 1],
        labels=["North-facing", "South-facing"]
    )
    ax.set(
        xlabel="Aspect",
        ylabel="Burn severity (RdNBR)"
    )
    fig.tight_layout()

    # 3. RdNBR distribution across profile-curvature classes.
    curvature_order = ["HN", "LN", "P", "LP", "HP"]
    curvature_names = [
        "HConcave", "MConcave", "Linear",
        "MConvex", "HConvex"
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.violinplot(
        data=data,
        x="ProfCurvCls1",
        y="RdNBR",
        order=curvature_order,
        ax=ax
    )

    ax.set_xticks(
        range(len(curvature_names)),
        labels=curvature_names
    )
    ax.set(
        xlabel="Profile curvature",
        ylabel="Burn severity (RdNBR)"
    )
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
