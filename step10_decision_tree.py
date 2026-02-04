import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
X_train = pd.read_csv("X_train.csv")
X_test  = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").values.ravel()
y_test  = pd.read_csv("y_test.csv").values.ravel()

print("Train-test data loaded")

# Handle inf and NaN
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

print("Missing values handled")

# Decision Tree model
dt = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=50,
    min_samples_leaf=25,
    random_state=42
)

dt.fit(X_train, y_train)
print("Decision Tree training completed")

# Prediction
y_pred = dt.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
