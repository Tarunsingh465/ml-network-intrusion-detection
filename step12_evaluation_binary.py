# =========================================================
# STEP 12: Phase 4.1 – Binary Evaluation (Benign vs Attack)
# Project: ML-Based Network Intrusion Detection System
# Dataset: CICIDS 2017
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)


# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv("cicids2017_clean.csv")
print("Dataset loaded successfully")
print("Original shape:", df.shape)


# 2. Handle Infinite & Missing Values
print("\nCleaning infinite and missing values...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)
print("Data cleaning completed")

# ---------------------------------------------------------
# 3. CORRECT BINARY LABEL MAPPING
# BENIGN -> 0
# ANY ATTACK -> 1
# ---------------------------------------------------------
print("\nConverting multiclass labels to binary...")

df["Binary_Label"] = df["Label"].apply(
    lambda x: 0 if x == "BENIGN" else 1
)

print("Binary label distribution:")
print(df["Binary_Label"].value_counts())


# 4. Features & Target
X = df.drop(columns=["Label", "Binary_Label"])
y = df["Binary_Label"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# 6. Train Random Forest Model
print("\nTraining Random Forest model...")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("Random Forest training completed")

# 7. Predictions
print("\nGenerating predictions...")

y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# 8. Confusion Matrix
print("\nConfusion Matrix:")

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Attack"],
    yticklabels=["Benign", "Attack"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Binary IDS")
plt.tight_layout()
plt.show()

# 9. Classification Report
print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Benign", "Attack"],
        zero_division=0
    )
)

# 10. ROC-AUC & ROC Curve
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC-AUC Score:", roc_auc)

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Binary IDS")
plt.legend()
plt.tight_layout()
plt.show()

# END
print("\nPHASE 4.1 (BINARY EVALUATION) COMPLETED SUCCESSFULLY")
