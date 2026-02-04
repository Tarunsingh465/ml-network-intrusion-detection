import pandas as pd

X = pd.read_csv("X_train.csv")

leak_cols = [col for col in X.columns if "label" in col.lower()]

print("Columns containing 'label':", leak_cols)
