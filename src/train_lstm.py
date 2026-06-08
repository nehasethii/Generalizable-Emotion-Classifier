import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import class_weight
import joblib

# ============================
# Paths
# ============================
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models/lstm"
RESULTS_DIR = "results/lstm"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(RESULTS_DIR, "lstm_results.csv")

# ============================
# Hyperparameters
# ============================
LEARNING_RATES = [0.01, 0.001, 0.0001]
BATCH_SIZES = [32, 64]
EPOCHS = 10

MAX_WORDS = 30000
MAX_LEN = 100
EMBED_DIM = 128
LSTM_UNITS = 128
DROPOUT = 0.3

# ============================
# Load Data
# ============================
train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "goemotions_train_clean.csv"))
val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "goemotions_val_clean.csv"))

train_texts = train_df["clean_text"].astype(str).tolist()
val_texts = val_df["clean_text"].astype(str).tolist()

# Encode labels
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["emotion"])
y_val = label_encoder.transform(val_df["emotion"])

joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

# ============================
# Tokenizer
# ============================
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

joblib.dump(tokenizer, os.path.join(MODEL_DIR, "tokenizer.pkl"))

X_train = pad_sequences(tokenizer.texts_to_sequences(train_texts), maxlen=MAX_LEN)
X_val = pad_sequences(tokenizer.texts_to_sequences(val_texts), maxlen=MAX_LEN)

# ============================
# Class Weights
# ============================
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# ============================
# Build Model
# ============================
def build_model(lr):
    model = Sequential([
        Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
        LSTM(LSTM_UNITS),
        Dropout(DROPOUT),
        Dense(LSTM_UNITS//2, activation="relu"),
        Dropout(0.3),
        Dense(len(label_encoder.classes_), activation="softmax")
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=["accuracy"]
    )
    return model

# ============================
# Training Loop
# ============================
all_results = []

for lr in LEARNING_RATES:
    for bs in BATCH_SIZES:

        print(f"\n🟩 Training LSTM | LR={lr} | BS={bs}")

        model = build_model(lr)

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            batch_size=bs,
            class_weight=class_weights,
            verbose=1
        )

        # Evaluation
        y_pred = np.argmax(model.predict(X_val), axis=1)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_val, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        # Save Model
        model.save(os.path.join(MODEL_DIR, f"lstm_lr{lr}_bs{bs}.h5"))

        # Add to CSV results
        all_results.append({
            "learning_rate": lr,
            "batch_size": bs,
            "epochs": EPOCHS,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

# ============================
# Save final CSV
# ============================
results_df = pd.DataFrame(all_results)
results_df.to_csv(RESULTS_CSV, index=False)

print("\n=============================================")
print("✅ Training Complete")
print(f"📁 Combined CSV saved: {RESULTS_CSV}")
print("=============================================\n")
