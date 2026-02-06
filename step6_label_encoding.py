import pandas as pd

df = pd.read_csv("Tuesday-WorkingHours.pcap_ISCX.csv", low_memory=False)
df.columns = df.columns.str.strip()

df['Label_encoded'] = df['Label'].apply(
    lambda x: 0 if x == 'BENIGN' else 1
)

print(df[['Label', 'Label_encoded']].head())
print(df['Label_encoded'].value_counts())