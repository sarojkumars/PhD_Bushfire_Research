
"""Utilities for topographic severity classification from prepared tabular data."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def load_data():
    path = Path(
        os.environ.get("TOPO_INPUT_CSV", "topography_model_input.csv")
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Set TOPO_INPUT_CSV to the prepared model table: {path}"
        )

    data = pd.read_csv(path)

    if data.shape[1] < 22:
        raise ValueError(
            "Expected at least 22 columns: "
            "18 predictors and four one-hot severity classes."
        )

    data = data.iloc[:, :22].dropna()

    if len(data) == 0:
        raise ValueError("No complete records are available.")

    # First 18 columns: topographic predictors.
    features = data.iloc[:, :18].apply(
        pd.to_numeric, errors="raise"
    )

    # Columns 19–22: one-hot encoded fire-severity classes.
    encoded = data.iloc[:, 18:22].apply(
        pd.to_numeric, errors="raise"
    ).to_numpy()

    if not (
        np.isin(encoded, [0, 1]).all()
        and (encoded.sum(axis=1) == 1).all()
    ):
        raise ValueError(
            "Target columns 19–22 must contain "
            "exactly one active class per row."
        )

    labels = encoded.argmax(axis=1)

    # Stratified random 80/20 train/test split.
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    return x_train, x_test, y_train, y_test
