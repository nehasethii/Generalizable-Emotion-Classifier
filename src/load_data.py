import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset (change sep="\t" to sep="," depending on your file)
df = pd.read_csv("data/raw/go_emotions.csv", sep="\t")

print("Full dataset shape:", df.shape)
print(df.head())

# Split into train/validation/test (80/10/10)
train, temp = train_test_split(df, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save splits
train.to_csv("data/raw/goemotions_train.csv", index=False)
val.to_csv("data/raw/goemotions_val.csv", index=False)
test.to_csv("data/raw/goemotions_test.csv", index=False)

print("✅ Split into train/val/test and saved in data/raw/")
