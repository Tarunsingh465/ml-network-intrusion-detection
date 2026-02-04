import pandas as pd

# Load original dataset
df = pd.read_csv("Tuesday-WorkingHours.pcap_ISCX.csv", low_memory=False)

# Clean column names
df.columns = df.columns.str.strip()

# Label encoding
df['Label_encoded'] = df['Label'].apply(
    lambda x: 0 if x == 'BENIGN' else 1
)

# Save clean dataset
df.to_csv("cicids2017_clean.csv", index=False)

print("✅ Clean dataset saved as cicids2017_clean.csv")
print("Final shape:", df.shape)
