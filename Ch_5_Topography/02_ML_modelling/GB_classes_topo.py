
"""Topographic predictors of fire-severity classes."""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score

from topo_data import load_data


def main():
    x_train, x_test, y_train, y_test = load_data()

    # Hyperparameter tuning:
    # Model configurations were evaluated manually rather than
    # through an automated GridSearchCV procedure.
    #
    # Candidate values (EXAMPLES — replace with actual tested values):
    # n_estimators:      [100, 200, 300, 400, 500]
    # max_depth:         [3, 5, 7, 10]
    # learning_rate:     [0.01, 0.05, 0.1]
    # min_samples_split: [2, 3, 5]
    #
    # Selected configuration used in this script:
    # n_estimators=500
    # max_depth=7
    # learning_rate=0.1
    # min_samples_split=3
    #
    # Note: The candidate values above are illustrative until
    # confirmed against the original experiment records.

    model = GradientBoostingClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.1,
        min_samples_split=3,
        random_state=42,
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    print(
        "GB test accuracy:",
        accuracy_score(y_test, predictions)
    )

    print(
        "GB 2-fold training CV accuracy:",
        np.mean(
            cross_val_score(
                model, x_train, y_train, cv=2
            )
        )
    )

    print(
        "Confusion matrix:\n",
        confusion_matrix(y_test, predictions)
    )

    print(
        classification_report(
            y_test, predictions, zero_division=0
        )
    )

    order = np.argsort(model.feature_importances_)

    plt.barh(
        x_train.columns[order],
        model.feature_importances_[order]
    )
    plt.xlabel("Relative importance")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
