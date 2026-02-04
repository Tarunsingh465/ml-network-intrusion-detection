import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned CICIDS 2017 dataset
df = pd.read_csv("cicids2017_clean.csv")

print("Dataset loaded successfully")
print("Dataset shape:", df.shape)

# Separate features and target
X = df.drop(["Label", "Label_encoded"], axis=1)
y = df["Label_encoded"]

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# Train-test split (80-20 with stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train-test split completed")

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

# Save split data for reuse
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Train and test CSV files saved successfully")
