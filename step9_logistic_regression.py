import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# Load train-test data
X_train = pd.read_csv("X_train.csv")
X_test  = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").values.ravel()
y_test  = pd.read_csv("y_test.csv").values.ravel()

print("Train-test data loaded")

# ==============================
# HANDLE INF AND NaN VALUES
# ==============================

# Replace inf with NaN
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

print("Infinity values replaced with NaN")

# Impute NaN values with mean
imputer = SimpleImputer(strategy="mean")

X_train = imputer.fit_transform(X_train)
X_test  = imputer.transform(X_test)

print("NaN values handled using mean imputation")


# LOGISTIC REGRESSION MODEL

lr = LogisticRegression(
    max_iter=1000,
    n_jobs=-1
)

lr.fit(X_train, y_train)
print("Logistic Regression training completed")

# Predictions
y_pred = lr.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
