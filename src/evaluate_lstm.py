# src/evaluate_lstm.py
import os, joblib, json
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from src.rules import apply_rules
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "models", "lstm")
TEST_CSV = os.path.join(ROOT, "data", "processed", "goemotions_test_clean.csv")
RESULTS_DIR = os.path.join(ROOT, "results", "lstm")
os.makedirs(RESULTS_DIR, exist_ok=True)

def find_file(directory, patterns):
    import fnmatch
    for root, _, files in os.walk(directory):
        for p in patterns:
            for f in fnmatch.filter(files, p):
                return os.path.join(root, f)
    return None

def load():
    h5 = find_file(MODEL_DIR, ["*.h5"])
    tok_p = find_file(MODEL_DIR, ["tokenizer*.pkl","tokenizer.pkl"])
    le_p = find_file(MODEL_DIR, ["label_encoder*.pkl","label_encoder.pkl"])
    if not h5 or not tok_p or not le_p:
        raise FileNotFoundError("LSTM artifacts not all present in " + MODEL_DIR)
    model = load_model(h5)
    tokenizer = joblib.load(tok_p)
    le = joblib.load(le_p)
    return model, tokenizer, le

def evaluate():
    model, tokenizer, le = load()
    df = pd.read_csv(TEST_CSV).dropna(subset=["clean_text","emotion"])
    texts = df["clean_text"].astype(str).tolist()
    y_true = df["emotion"].tolist()
    # tokenization
    seqs = tokenizer.texts_to_sequences([t for t in texts])
    maxlen = model.input_shape[1] if model.input_shape and model.input_shape[1] else 100
    X = pad_sequences(seqs, maxlen=maxlen)
    preds = []
    for i, t in enumerate(texts):
        rlabel, rule = apply_rules(t)
        if rlabel:
            preds.append(rlabel); continue
    # model preds (batch)
    probs = model.predict(X, batch_size=128)
    idxs = np.argmax(probs, axis=1)
    pred_labels = le.inverse_transform(idxs)
    # merge rule overrides: replace those entries
    final_preds = []
    ip=0
    for i,t in enumerate(texts):
        rlabel, rule = apply_rules(t)
        if rlabel:
            final_preds.append(rlabel)
        else:
            final_preds.append(pred_labels[ip]); ip+=1

    acc = accuracy_score(y_true, final_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, final_preds, average="weighted", zero_division=0)
    report = classification_report(y_true, final_preds, zero_division=0)

    out = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "report": report}
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    df_out = df.copy(); df_out["pred"] = final_preds
    df_out.to_csv(os.path.join(RESULTS_DIR, f"lstm_test_preds_{ts}.csv"), index=False, encoding="utf-8")
    with open(os.path.join(RESULTS_DIR, f"lstm_metrics_{ts}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("LSTM metrics:", out)
    return out

if __name__ == "__main__":
    evaluate()
