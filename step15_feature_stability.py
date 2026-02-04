# =========================================================
# STEP 15: Phase 4.2.3 – Feature Importance Stability
# =========================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# 1. Load Dataset
df = pd.read_csv("cicids2017_clean.csv")

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# Binary label mapping
df["Binary_Label"] = df["Label"].apply(
    lambda x: 0 if x == "BENIGN" else 1
)

X = df.drop(columns=["Label", "Binary_Label", "Label_encoded"], errors="ignore")
y = df["Binary_Label"]

# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# 4. Feature Importance
importances = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# 5. Show Top 15 Features
print("\nTop 15 Important Features:\n")
print(feature_importance_df.head(15))

print("\nSTEP 4.2.3 COMPLETED")
