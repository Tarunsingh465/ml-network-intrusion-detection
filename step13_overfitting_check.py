# STEP 13: Phase 4.2.1 – Overfitting Check

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

# 1. Load Dataset
df = pd.read_csv("cicids2017_clean.csv")

# Clean infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# Binary label mapping
df["Binary_Label"] = df["Label"].apply(
    lambda x: 0 if x == "BENIGN" else 1
)

X = df.drop(columns=["Label", "Binary_Label"])
y = df["Binary_Label"]

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Train Model
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# 4. Predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# 5. Metrics Comparison
print("\nTRAIN PERFORMANCE")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("Attack Recall:", recall_score(y_train, y_train_pred))

print("\nTEST PERFORMANCE")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Attack Recall:", recall_score(y_test, y_test_pred))

print("\nSTEP 4.2.1 COMPLETED")
