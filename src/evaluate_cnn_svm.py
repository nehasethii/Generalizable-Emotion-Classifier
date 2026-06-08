# src/evaluate_cnn_svm.py
import os, joblib, json
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from src.rules import apply_rules
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "models", "cnn_svm")
TEST_CSV = os.path.join(ROOT, "data", "processed", "goemotions_test_clean.csv")
RESULTS_DIR = os.path.join(ROOT, "results", "cnn_svm")
os.makedirs(RESULTS_DIR, exist_ok=True)

def find_file(directory, patterns):
    import fnmatch
    for root, _, files in os.walk(directory):
        for p in patterns:
            for f in fnmatch.filter(files, p):
                return os.path.join(root, f)
    return None

def load():
    enc_p = find_file(MODEL_DIR, ["*encoder*.h5","cnn_encoder*.h5","*.h5"])
    tok_p = find_file(MODEL_DIR, ["tokenizer*.json","tokenizer.json"])
    svm_p = find_file(MODEL_DIR, ["*svm*.joblib","*.joblib","*.pkl"])
    le_p  = find_file(MODEL_DIR, ["label_encoder*.joblib","label_encoder.joblib"])
    if not enc_p or not tok_p or not svm_p or not le_p:
        raise FileNotFoundError("CNN+SVM artifacts missing in " + MODEL_DIR)
    encoder = load_model(enc_p)
    import json as _json
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    with open(tok_p, "r", encoding="utf-8") as fh:
        tok = tokenizer_from_json(fh.read())
    svm = joblib.load(svm_p)
    le = joblib.load(le_p)
    return encoder, tok, svm, le

def evaluate():
    encoder, tok, svm, le = load()
    df = pd.read_csv(TEST_CSV).dropna(subset=["clean_text","emotion"])
    texts = df["clean_text"].astype(str).tolist()
    y_true = df["emotion"].tolist()
    maxlen = encoder.input_shape[1] if encoder.input_shape and encoder.input_shape[1] else 100
    seqs = tok.texts_to_sequences(texts)
    X = pad_sequences(seqs, maxlen=maxlen)
    feats = encoder.predict(X, batch_size=128)
    if hasattr(svm, "predict_proba"):
        probs = svm.predict_proba(feats)
        idxs = np.argmax(probs, axis=1)
        pred_labels = le.inverse_transform(idxs)
    else:
        idxs = svm.predict(feats)
        pred_labels = le.inverse_transform(idxs)
    # rules override
    final_preds = []
    for t, pl in zip(texts, pred_labels):
        rlabel, rule = apply_rules(t)
        final_preds.append(rlabel if rlabel else pl)

    acc = accuracy_score(y_true, final_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, final_preds, average="weighted", zero_division=0)
    report = classification_report(y_true, final_preds, zero_division=0)
    out = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "report": report}
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    df_out = df.copy(); df_out["pred"] = final_preds
    df_out.to_csv(os.path.join(RESULTS_DIR, f"cnn_svm_test_preds_{ts}.csv"), index=False, encoding="utf-8")
    with open(os.path.join(RESULTS_DIR, f"cnn_svm_metrics_{ts}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("CNN+SVM metrics:", out)
    return out

if __name__ == "__main__":
    evaluate()
