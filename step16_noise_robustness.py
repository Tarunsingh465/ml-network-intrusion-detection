# =========================================================
# STEP 16: Phase 4.2.4 – Noise Robustness Test
# =========================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score

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

# 3. Train Model
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# 4. Baseline Recall (No Noise)
y_pred_clean = rf_model.predict(X_test)
baseline_recall = recall_score(y_test, y_pred_clean)

print("Baseline Attack Recall (no noise):", baseline_recall)

# 5. Add Gaussian Noise to Test Data
np.random.seed(42)

noise_factor = 0.01  # 1% noise
noise = noise_factor * np.random.normal(size=X_test.shape)

X_test_noisy = X_test + noise

# 6. Recall After Noise
y_pred_noisy = rf_model.predict(X_test_noisy)
noisy_recall = recall_score(y_test, y_pred_noisy)

print("Attack Recall after noise:", noisy_recall)

print("\nSTEP 4.2.4 COMPLETED")
