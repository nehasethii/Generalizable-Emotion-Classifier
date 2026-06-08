import pandas as pd

df = pd.read_csv("data/processed/goemotions_train_clean.csv")

print("Columns:\n", df.columns.tolist(), "\n")
print("First few rows:")
print(df.head(3))
print("\nUnique values in the first few label columns:")
for col in df.columns[8:15]:  # adjust range if needed
    print(f"{col}: {df[col].unique()[:5]}")
