
# Chapter 4 — Random Forest feature importance
# Meteorological predictors of fire severity

import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Load data
fd = pd.read_csv(
    os.environ.get("METEO_INPUT_CSV", "data/model_input.csv")
)

print(fd.head())

# Predictor and response variables
fd_x = fd.iloc[:, 3:65]
fd_y = fd.iloc[:, 65]

print(fd_x.head())
print(fd_y.head())


# Training and test split
X_train, X_test, y_train, y_test = train_test_split(
    fd_x,
    fd_y,
    test_size=0.2,
    random_state=42
)


# Random Forest hyperparameter tuning
# Alternative candidate values:
# n_estimators: 100, 200, 300, 500
# max_depth: 5, 10, 20, None
# min_samples_split: 2, 5, 10
# min_samples_leaf: 1, 2, 4
# max_features: "sqrt", "log2", None
# criterion: "gini", "entropy"

rfc = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)


# Model training
rfc.fit(X_train, y_train)


# Predictions
y_pred_train = rfc.predict(X_train)
y_pred_test = rfc.predict(X_test)


# Model evaluation
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Training accuracy:", round(train_accuracy, 4))
print("Test accuracy:", round(test_accuracy, 4))


# Feature importance
feature_importance_df = pd.DataFrame({
    "feature": fd_x.columns,
    "importance": rfc.feature_importances_
})

feature_importance_df = feature_importance_df.sort_values(
    by="importance",
    ascending=False
)

print(feature_importance_df)


# Export feature importance
feature_importance_df.to_csv(
    "RF_feature_importance.csv",
    index=False
)
