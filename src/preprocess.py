import pandas as pd
import re
import os

# Paths
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def clean_text(text):
    """Basic text cleaning."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(file_path, save_path):
    """Clean text and save processed data."""
    df = pd.read_csv(file_path)

    # If 'text' column exists, rename to maintain consistency
    if "text" in df.columns and "clean_text" not in df.columns:
        df.rename(columns={"text": "clean_text"}, inplace=True)

    # Handle missing values
    df["clean_text"] = df["clean_text"].fillna("")

    # Clean text
    df["clean_text"] = df["clean_text"].apply(clean_text)

    # Drop rows with missing emotion labels (if any)
    if "emotion" in df.columns:
        df = df.dropna(subset=["emotion"])

    print(f"Processed {len(df)} samples from {file_path}")
    print("Sample rows:\n", df.head(), "\n")

    df.to_csv(save_path, index=False)
    print(f"✅ Saved processed file to {save_path}\n")

# Run preprocessing for train, val, and test
preprocess(
    os.path.join(RAW_DIR, "goemotions_train.csv"),
    os.path.join(PROCESSED_DIR, "goemotions_train_clean.csv")
)
preprocess(
    os.path.join(RAW_DIR, "goemotions_val.csv"),
    os.path.join(PROCESSED_DIR, "goemotions_val_clean.csv")
)
preprocess(
    os.path.join(RAW_DIR, "goemotions_test.csv"),
    os.path.join(PROCESSED_DIR, "goemotions_test_clean.csv")
)
