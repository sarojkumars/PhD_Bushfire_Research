# =============================================================================
# 13_spatial_sampling_and_autocorrelation.py
#
# Appendix A - Spatial sampling and spatial autocorrelation sensitivity
#
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.spatial import cKDTree
from appendix_config import SPATIAL_DATA_FILE as DATA_FILE, SPATIAL_DIR as OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Distance classes in coordinate units
## -----------------------------------------------------------------------------

DISTANCE_BINS = [
    (400, 500),
    (500, 750),
    (750, 1000),
    (1000, 1500),
    (1500, 2000),
    (2000, 3000),
    (3000, 5000)
]


# Variables to analyse
MORAN_VARIABLES = [
    "SNIC2clsN",
    "RdNBR"
]


# =============================================================================
# 2. READ DATA
# =============================================================================

print("=" * 75)
print("APPENDIX A SPATIAL SAMPLING AND AUTOCORRELATION")
print("=" * 75)

df = pd.read_csv(DATA_FILE)

print(f"\nObservations: {len(df):,}")
print(f"Columns:      {len(df.columns):,}")


required = [
    "pt_id",
    "x_cood",
    "y_cood",
    "SNIC2clsN",
    "RdNBR"
]

missing_columns = [
    x for x in required
    if x not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# =============================================================================
# 3. DATA INTEGRITY
# =============================================================================

print("\n" + "=" * 75)
print("COORDINATE INTEGRITY")
print("=" * 75)

print(
    "Missing X:",
    df["x_cood"].isna().sum()
)

print(
    "Missing Y:",
    df["y_cood"].isna().sum()
)

print(
    "Duplicate pt_id:",
    df["pt_id"].duplicated().sum()
)


if (
    df["x_cood"].isna().any()
    or df["y_cood"].isna().any()
):
    raise ValueError(
        "Missing coordinates detected."
    )


if df["pt_id"].duplicated().any():
    raise ValueError(
        "Duplicate pt_id detected."
    )


duplicate_xy = df.duplicated(
    subset=["x_cood", "y_cood"],
    keep=False
)

print(
    "Rows sharing identical coordinates:",
    duplicate_xy.sum()
)


# =============================================================================
# 4. COORDINATE ARRAY + KD TREE
# =============================================================================

coords = df[
    ["x_cood", "y_cood"]
].to_numpy(dtype=np.float64)

print("\nBuilding spatial KD-tree...")

tree = cKDTree(coords)


# =============================================================================
# 5. NEAREST-NEIGHBOUR ANALYSIS
# =============================================================================

print("\n" + "=" * 75)
print("NEAREST-NEIGHBOUR ANALYSIS")
print("=" * 75)

distances, indices = tree.query(
    coords,
    k=2,
    workers=-1
)

nn_distance = distances[:, 1]
nn_index = indices[:, 1]


nn_results = pd.DataFrame({

    "pt_id":
        df["pt_id"].values,

    "nearest_pt_id":
        df.iloc[nn_index]["pt_id"].values,

    "nearest_distance":
        nn_distance
})


nn_results.to_csv(
    OUTPUT_DIR /
    "spatial_nearest_neighbour_distances.csv",
    index=False
)


print(
    f"Minimum: {nn_distance.min():.3f}"
)

print(
    f"Mean:    {nn_distance.mean():.3f}"
)

print(
    f"Median:  {np.median(nn_distance):.3f}"
)

print(
    f"Maximum: {nn_distance.max():.3f}"
)


# =============================================================================
# 6. SAMPLING-DISTANCE CHECK
# =============================================================================

thresholds = [
    100,
    200,
    300,
    350,
    375,
    390,
    399,
    400,
    401,
    425,
    450,
    500,
    750,
    1000
]


spacing_records = []


print("\n" + "=" * 75)
print("SAMPLING-DISTANCE CHECK")
print("=" * 75)


for threshold in thresholds:

    n = int(
        np.sum(
            nn_distance < threshold
        )
    )

    pct = (
        n /
        len(nn_distance) *
        100
    )

    spacing_records.append({

        "threshold":
            threshold,

        "n_below_threshold":
            n,

        "percent_below_threshold":
            pct
    })

    print(
        f"< {threshold:4d}: "
        f"{n:7,d} "
        f"({pct:7.3f}%)"
    )


spacing_df = pd.DataFrame(
    spacing_records
)


spacing_df.to_csv(
    OUTPUT_DIR /
    "spatial_spacing_threshold_summary.csv",
    index=False
)


# =============================================================================
# 7. NEAREST-NEIGHBOUR FIGURE
# =============================================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.hist(
    nn_distance,
    bins=100
)

ax.axvline(
    400,
    linestyle="--",
    linewidth=1.5,
    label="400"
)

ax.set_xlabel(
    "Nearest-neighbour distance"
)

ax.set_ylabel(
    "Number of observations"
)

ax.set_title(
    "Nearest-neighbour distances in the analytical sample"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR /
    "Fig_spatial_nearest_neighbour_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# =============================================================================
# 8. ECDF
# =============================================================================

sorted_nn = np.sort(
    nn_distance
)

ecdf = (
    np.arange(
        1,
        len(sorted_nn) + 1
    )
    /
    len(sorted_nn)
)


fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    sorted_nn,
    ecdf
)

ax.axvline(
    400,
    linestyle="--",
    linewidth=1.5,
    label="400"
)

upper_limit = min(
    1500,
    np.quantile(
        sorted_nn,
        0.995
    )
)

ax.set_xlim(
    0,
    upper_limit
)

ax.set_xlabel(
    "Nearest-neighbour distance"
)

ax.set_ylabel(
    "Cumulative proportion"
)

ax.set_title(
    "Cumulative nearest-neighbour distance distribution"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR /
    "Fig_spatial_nearest_neighbour_ECDF.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# =============================================================================
# 9. BUILD ALL NEIGHBOUR PAIRS WITHIN 5 KM
# =============================================================================
#
# =============================================================================

MAX_DISTANCE = max(
    upper
    for lower, upper
    in DISTANCE_BINS
)


print("\n" + "=" * 75)
print("BUILDING SPATIAL NEIGHBOUR PAIRS")
print("=" * 75)

print(
    f"Maximum distance: {MAX_DISTANCE}"
)

print(
    "Finding point pairs..."
)


pairs = tree.query_pairs(
    r=MAX_DISTANCE,
    output_type="ndarray"
)


print(
    f"Total pairs within {MAX_DISTANCE}: "
    f"{len(pairs):,}"
)


# =============================================================================
# 10. CALCULATE DISTANCE FOR EACH PAIR
# =============================================================================

print(
    "Calculating pair distances..."
)

dx = (
    coords[pairs[:, 0], 0]
    -
    coords[pairs[:, 1], 0]
)

dy = (
    coords[pairs[:, 0], 1]
    -
    coords[pairs[:, 1], 1]
)

pair_distance = np.sqrt(
    dx**2 + dy**2
)


# =============================================================================
# 11. MORAN'S I FUNCTION
# =============================================================================
## This corresponds to binary symmetric distance-class weights.
# =============================================================================

def moran_from_pairs(
    values,
    pair_array
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    valid = np.isfinite(values)



    if not np.all(valid):

        mapping = np.full(
            len(values),
            -1,
            dtype=np.int64
        )

        mapping[valid] = np.arange(
            valid.sum()
        )

        keep_pairs = (
            valid[pair_array[:, 0]]
            &
            valid[pair_array[:, 1]]
        )

        p = pair_array[
            keep_pairs
        ]

        p = np.column_stack([
            mapping[p[:, 0]],
            mapping[p[:, 1]]
        ])

        x = values[valid]

    else:

        p = pair_array
        x = values


    n = len(x)

    if len(p) == 0:

        return np.nan


    z = x - np.mean(x)


    denominator = np.sum(
        z**2
    )


    if denominator == 0:

        return np.nan


    numerator = np.sum(
        z[p[:, 0]]
        *
        z[p[:, 1]]
    )


    I = (
        n /
        len(p)
        *
        numerator /
        denominator
    )


    return I


# =============================================================================
# 12. DISTANCE-CLASS MORAN'S I
# =============================================================================

print("\n" + "=" * 75)
print("DISTANCE-CLASS MORAN'S I")
print("=" * 75)


results = []


for variable in MORAN_VARIABLES:

    values = pd.to_numeric(
        df[variable],
        errors="coerce"
    ).to_numpy()


    print(
        f"\nVariable: {variable}"
    )


    for lower, upper in DISTANCE_BINS:

        # Lower inclusive
        # Upper exclusive
        mask = (
            (pair_distance >= lower)
            &
            (pair_distance < upper)
        )


        bin_pairs = pairs[
            mask
        ]


        I = moran_from_pairs(
            values,
            bin_pairs
        )


        # Number of unique observations
        # participating in this distance class

        if len(bin_pairs) > 0:

            participating = np.unique(
                bin_pairs.ravel()
            )

            n_points = len(
                participating
            )

        else:

            n_points = 0


        midpoint = (
            lower + upper
        ) / 2


        result = {

            "variable":
                variable,

            "distance_lower":
                lower,

            "distance_upper":
                upper,

            "distance_midpoint":
                midpoint,

            "n_pairs":
                len(bin_pairs),

            "n_points_with_neighbour":
                n_points,

            "moran_I":
                I
        }


        results.append(
            result
        )


        print(
            f"{lower:5.0f} - "
            f"{upper:5.0f}: "
            f"I = {I: .6f}, "
            f"pairs = {len(bin_pairs):,}, "
            f"points = {n_points:,}"
        )


moran_df = pd.DataFrame(
    results
)


moran_df.to_csv(
    OUTPUT_DIR /
    "spatial_moran_distance_classes.csv",
    index=False
)


# =============================================================================
# 13. EXPECTED MORAN'S I UNDER RANDOM SPATIAL ARRANGEMENT
# =============================================================================
#
# =============================================================================

expected_I = (
    -1 /
    (len(df) - 1)
)


print(
    "\nExpected Moran's I under randomisation:"
)

print(
    f"{expected_I:.8f}"
)


# =============================================================================
# 14. CORRELOGRAM FIGURE - ONE FIGURE PER VARIABLE
# =============================================================================

for variable in MORAN_VARIABLES:

    plot_df = (
        moran_df[
            moran_df["variable"]
            ==
            variable
        ]
        .copy()
    )


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    ax.plot(
        plot_df[
            "distance_midpoint"
        ],
        plot_df[
            "moran_I"
        ],
        marker="o"
    )


    ax.axhline(
        expected_I,
        linestyle="--",
        linewidth=1
    )


    ax.axhline(
        0,
        linewidth=0.8
    )


    ax.set_xlabel(
        "Distance-class midpoint"
    )


    ax.set_ylabel(
        "Moran's I"
    )


    if variable == "SNIC2clsN":

        title = (
            "Spatial correlogram of binary fire severity"
        )

    else:

        title = (
            "Spatial correlogram of continuous fire severity"
        )


    ax.set_title(
        title
    )


    fig.tight_layout()


    fig.savefig(
        OUTPUT_DIR /
        f"Fig_Moran_correlogram_{variable}.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close(fig)


# =============================================================================
# 15. COMBINED SUMMARY TABLE
# =============================================================================

summary_table = moran_df.pivot(
    index=[
        "distance_lower",
        "distance_upper",
        "distance_midpoint"
    ],
    columns="variable",
    values="moran_I"
).reset_index()


summary_table.to_csv(
    OUTPUT_DIR /
    "Table_spatial_Moran_summary.csv",
    index=False
)


# =============================================================================
# 16. SAVE TEXT SUMMARY
# =============================================================================

summary_file = (
    OUTPUT_DIR /
    "Spatial_Autocorrelation_Summary.txt"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Appendix A spatial sampling and "
        "autocorrelation analysis\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )


    f.write(
        f"Observations: {len(df):,}\n"
    )

    f.write(
        f"Minimum nearest-neighbour distance: "
        f"{nn_distance.min():.6f}\n"
    )

    f.write(
        f"Mean nearest-neighbour distance: "
        f"{nn_distance.mean():.6f}\n"
    )

    f.write(
        f"Median nearest-neighbour distance: "
        f"{np.median(nn_distance):.6f}\n"
    )

    f.write(
        f"Maximum nearest-neighbour distance: "
        f"{nn_distance.max():.6f}\n\n"
    )


    f.write(
        "Distance-class Moran's I\n"
    )

    f.write(
        "-" * 70 + "\n"
    )


    for _, row in moran_df.iterrows():

        f.write(
            f"{row['variable']:12s} "
            f"{row['distance_lower']:6.0f}-"
            f"{row['distance_upper']:6.0f} "
            f"I={row['moran_I']: .6f} "
            f"pairs={int(row['n_pairs']):,} "
            f"points="
            f"{int(row['n_points_with_neighbour']):,}\n"
        )


    f.write(
        "\nExpected Moran's I under randomisation: "
        f"{expected_I:.8f}\n"
    )


# =============================================================================
# 17. FINISH
# =============================================================================

print("\n" + "=" * 75)
print("SPATIAL ANALYSIS COMPLETE")
print("=" * 75)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR
)

print(
    "\nMain result table:"
)

print(
    OUTPUT_DIR /
    "spatial_moran_distance_classes.csv"
)

print(
    "\nCorrelogram figures:"
)

print(
    OUTPUT_DIR /
    "Fig_Moran_correlogram_SNIC2clsN.png"
)

print(
    OUTPUT_DIR /
    "Fig_Moran_correlogram_RdNBR.png"
)