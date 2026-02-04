# =========================================================
# STEP 14: Phase 4.2.2 – K-Fold Cross Validation
# =========================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------
df = pd.read_csv("cicids2017_clean.csv")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# Binary label mapping
df["Binary_Label"] = df["Label"].apply(
    lambda x: 0 if x == "BENIGN" else 1
)

X = df.drop(columns=["Label", "Binary_Label"])
y = df["Binary_Label"]

# ---------------------------------------------------------
# 2. K-Fold Cross Validation
# ---------------------------------------------------------
print("Starting 5-Fold Cross Validation...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(
    rf_model,
    X,
    y,
    cv=cv,
    scoring="recall"
)

# ---------------------------------------------------------
# 3. Results
# ---------------------------------------------------------
print("\nCross-validation Recall Scores:", scores)
print("Mean Recall:", scores.mean())
print("Standard Deviation:", scores.std())

print("\nSTEP 4.2.2 COMPLETED")
