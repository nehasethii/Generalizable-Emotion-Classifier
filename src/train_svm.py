# src/train_svm_fast_linear.py
import os, joblib
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from datetime import datetime

MODEL_DIR = "models/cnn_svm"
PROCESSED_DIR = "data/processed"
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")
# use latest encoder available
encoders = sorted([os.path.join(MODEL_DIR,f) for f in os.listdir(MODEL_DIR) if f.startswith("cnn_encoder")])
if not encoders:
    raise SystemExit("No cnn_encoder file found in models/cnn_svm")
ENCODER_PATH = encoders[-1]

print("Using encoder:", ENCODER_PATH)
tok = tokenizer_from_json(open(TOKENIZER_PATH).read())
enc = load_model(ENCODER_PATH)

train_df = pd.read_csv(os.path.join(PROCESSED_DIR,"goemotions_train_clean.csv"))
val_df = pd.read_csv(os.path.join(PROCESSED_DIR,"goemotions_val_clean.csv"))
train_df.dropna(subset=["clean_text","emotion"], inplace=True)
val_df.dropna(subset=["clean_text","emotion"], inplace=True)

X_train = pad_sequences(tok.texts_to_sequences(train_df["clean_text"].astype(str).tolist()), maxlen=100)
X_val   = pad_sequences(tok.texts_to_sequences(val_df["clean_text"].astype(str).tolist()), maxlen=100)
print("Extracting features (train)...")
feats_train = enc.predict(X_train, batch_size=128, verbose=1)
print("Extracting features (val)...")
feats_val   = enc.predict(X_val, batch_size=128, verbose=1)

y_train = train_df["emotion"].factorize()[0]
y_val   = val_df["emotion"].factorize()[0]

print("Training fast LogisticRegression (very quick)...")
clf = LogisticRegression(max_iter=2000, n_jobs=-1, class_weight="balanced")
t0 = datetime.now()
clf.fit(feats_train, y_train)
t1 = datetime.now()
print("Trained in", (t1-t0).total_seconds(), "s")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = os.path.join(MODEL_DIR, f"svm_fast_linear_{stamp}.joblib")
joblib.dump(clf, out_path)
print("Saved classifier:", out_path)

probs = clf.predict_proba(feats_val)
preds = probs.argmax(axis=1)
print("Accuracy:", accuracy_score(y_val,preds))
print("F1 (weighted):", f1_score(y_val,preds, average='weighted'))
# append result to CSV
res = {
    "run_time": datetime.now().isoformat(),
    "encoder_path": ENCODER_PATH,
    "clf": "LogisticRegression",
    "accuracy": float(accuracy_score(y_val,preds)),
    "f1": float(f1_score(y_val,preds, average='weighted'))
}
os.makedirs("results/cnn_svm", exist_ok=True)
csvp = "results/cnn_svm/cnn_svm_grid_results.csv"
if os.path.exists(csvp):
    df_prev = pd.read_csv(csvp)
    df_out = pd.concat([df_prev, pd.DataFrame([res])], ignore_index=True)
else:
    df_out = pd.DataFrame([res])
df_out.to_csv(csvp, index=False)
print("Appended results to", csvp)
