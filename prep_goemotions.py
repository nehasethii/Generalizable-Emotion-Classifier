import pandas as pd
import re
from sklearn.model_selection import train_test_split

# === Step 1: Load original data ===
df = pd.read_csv("data/raw/go_emotions.csv")
print("✅ Original data loaded. Shape:", df.shape)

# === Step 2: Clean text ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)       # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)     # keep only letters and spaces
    text = re.sub(r"\s+", " ", text).strip()  # normalize spaces
    return text

df["clean_text"] = df["text"].apply(clean_text)

# === Step 3: Identify emotion label columns ===
label_cols = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# === Step 4: Convert multi-label → single (clean_text, emotion) pairs ===
records = []
for _, row in df.iterrows():
    for label in label_cols:
        if row.get(label) == 1:
            records.append((row["clean_text"], label))

pairs_df = pd.DataFrame(records, columns=["clean_text", "emotion"])
print(f"✅ Created {len(pairs_df)} text-emotion pairs.")

# === Step 5: Remove duplicates & NaNs (just in case) ===
pairs_df.dropna(subset=["clean_text", "emotion"], inplace=True)
pairs_df.drop_duplicates(inplace=True)
print("✅ After cleaning:", pairs_df.shape)

# === Step 6: Split into train / val / test ===
train_df, temp_df = train_test_split(pairs_df, test_size=0.2, random_state=42, stratify=pairs_df["emotion"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["emotion"])

# === Step 7: Save ===
train_df.to_csv("data/processed/goemotions_train.csv", index=False)
val_df.to_csv("data/processed/goemotions_val.csv", index=False)
test_df.to_csv("data/processed/goemotions_test.csv", index=False)

print("\n✅ Splits created and saved:")
print("Train:", train_df.shape)
print("Val:", val_df.shape)
print("Test:", test_df.shape)

print("\n📊 Example rows:")
print(train_df.sample(10))
