
# Chapter 4 — XGBoost classification and feature importance
# Meteorological predictors of fire severity

import os

import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load data
fd = pd.read_csv(
    os.environ.get("METEO_INPUT_CSV", "data/model_input.csv")
)

print(fd.head())
print(fd.dtypes)

# Predictor and response variables
fd_x = fd.iloc[:, 3:65]
fd_y = fd.iloc[:, 65]

print(fd_x.head())


# Training and test split
X_train, X_test, y_train, y_test = train_test_split(
    fd_x,
    fd_y,
    test_size=0.20,
    random_state=42
)


# XGBoost hyperparameter tuning
# Alternative candidate values:
# learning_rate: 0.01, 0.1, 1.0
# n_estimators: 100, 200, 500, 1000
# max_depth: 3, 5, 7
# subsample: 0.7, 0.8, 1.0
# colsample_bytree: 0.7, 0.8, 1.0

xgb_Classifier = xgb.XGBClassifier(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=3,
    random_state=42
)


# Model training
xgb_Classifier.fit(X_train, y_train)


# Predictions
xgb_predict = xgb_Classifier.predict(X_test)


# Model evaluation
print(
    "Accuracy of Model:",
    accuracy_score(y_test, xgb_predict)
)


# Feature importance
fig, ax = plt.subplots(figsize=(12, 14))

xgb.plot_importance(
    xgb_Classifier,
    ax=ax
)

fig.tight_layout()
fig.savefig(
    "XGBClassifier_Feature_Importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
