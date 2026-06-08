import pandas as pd

# Load datasets
train = pd.read_csv("data/processed/goemotions_train_clean.csv")
val = pd.read_csv("data/processed/goemotions_val_clean.csv")

# Define emotion columns (update if needed)
label_cols = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
    'remorse', 'sadness', 'surprise', 'neutral'
]

def convert_to_single_label(df):
    # Only keep available emotion columns
    existing_labels = [col for col in label_cols if col in df.columns]
    
    # Ensure numeric labels
    df[existing_labels] = df[existing_labels].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    
    # Build clean text–emotion pairs
    pairs = []
    for _, row in df.iterrows():
        for emo in existing_labels:
            if row[emo] == 1:
                pairs.append((row["clean_text"], emo))
    return pd.DataFrame(pairs, columns=["clean_text", "emotion"])

# Convert
train_single = convert_to_single_label(train)
val_single = convert_to_single_label(val)

# Save
train_single.to_csv("data/processed/goemotions_train_single.csv", index=False)
val_single.to_csv("data/processed/goemotions_val_single.csv", index=False)

print("✅ Successfully created clean text-emotion mapping files!")
print(f"Train samples: {len(train_single)}, Val samples: {len(val_single)}")
print("\nSample rows:")
print(train_single.head())
