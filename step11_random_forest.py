import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Load data
X_train = pd.read_csv("X_train.csv")
joblib.dump(X_train.columns.tolist(), "model/feature_columns.pkl")
X_test  = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").values.ravel()
y_test  = pd.read_csv("y_test.csv").values.ravel()

print("Train-test data loaded")

print("FEATURE COLUMNS:")
print(X_train.columns.tolist())



# Handle NaN / Inf
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

print("Missing values handled")


# Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=42
)

rf_model.fit(X_train, y_train)
print("Random Forest training completed")


# Prediction
y_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# 🔹 SAVE MODEL 
joblib.dump(rf_model, "model/random_forest_model.pkl")

print("Model saved inside model folder")
