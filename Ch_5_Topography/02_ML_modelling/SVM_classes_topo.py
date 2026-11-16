
"""Topographic predictors of fire-severity classes."""

import numpy as np

from sklearn.svm import SVC
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
    # kernel: ["linear", "rbf", "poly", "sigmoid"]
    # C:      [0.1, 1, 10, 100, 1000]
    # gamma:  ["scale", "auto", 0.001, 0.01, 0.1]
    #
    # Selected configuration used in this script:
    # kernel="rbf"
    # C=1000.0
    # gamma="scale" (scikit-learn default)
    #
    # Note: Candidate values above are illustrative and should be
    # confirmed against the original experimental records.

    model = SVC(
        kernel="rbf",
        C=1000.0,
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    print(
        "SVM test accuracy:",
        accuracy_score(y_test, predictions)
    )

    print(
        "SVM 10-fold training CV accuracy:",
        np.mean(
            cross_val_score(
                model, x_train, y_train, cv=10
            )
        )
    )

    print(
        "Confusion matrix:\n",
        confusion_matrix(y_test, predictions)
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


if __name__ == "__main__":
    main()
