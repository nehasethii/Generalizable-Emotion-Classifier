# src/evaluate_distilbert.py
import os, json
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from src.rules import apply_rules
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# accept common distilbert folder names
for cand in ("models/DistilBERT", "models/distilbert", "models/distilbert_emotion"):
    if os.path.isdir(os.path.join(ROOT, cand)):
        DISTIL_DIR = os.path.join(ROOT, cand)
        break
else:
    DISTIL_DIR = None

TEST_CSV = os.path.join(ROOT, "data", "processed", "goemotions_test_clean.csv")
RESULTS_DIR = os.path.join(ROOT, "results", "distilbert")
os.makedirs(RESULTS_DIR, exist_ok=True)

def load():
    if DISTIL_DIR is None:
        raise FileNotFoundError("DistilBERT directory not found under models/")
    tokenizer = DistilBertTokenizerFast.from_pretrained(DISTIL_DIR, local_files_only=True)
    model = DistilBertForSequenceClassification.from_pretrained(DISTIL_DIR, local_files_only=True)
    id2label = getattr(model.config, "id2label", None)
    return tokenizer, model, id2label

def evaluate():
    tokenizer, model, id2label = load()
    df = pd.read_csv(TEST_CSV).dropna(subset=["clean_text","emotion"])
    texts = df["clean_text"].astype(str).tolist()
    y_true = df["emotion"].tolist()

    # batch inference
    batch_size = 64
    preds = []
    model.eval()
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc = {k:v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits.cpu().numpy()
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = probs / probs.sum(axis=1, keepdims=True)
            idxs = probs.argmax(axis=1)
            for j,idx in enumerate(idxs):
                lbl = id2label.get(str(idx)) if id2label else None
                if lbl is None:
                    try: lbl = model.config.id2label[idx]
                    except Exception: lbl = str(idx)
                preds.append(lbl)

    # apply rules override
    final_preds = []
    for t, p in zip(texts, preds):
        rlabel, rule = apply_rules(t)
        final_preds.append(rlabel if rlabel else p)

    acc = accuracy_score(y_true, final_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, final_preds, average="weighted", zero_division=0)
    report = classification_report(y_true, final_preds, zero_division=0)
    out = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "report": report}
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    df_out = df.copy(); df_out["pred"] = final_preds
    df_out.to_csv(os.path.join(RESULTS_DIR, f"distilbert_test_preds_{ts}.csv"), index=False, encoding="utf-8")
    with open(os.path.join(RESULTS_DIR, f"distilbert_metrics_{ts}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("DistilBERT metrics:", out)
    return out

if __name__ == "__main__":
    evaluate()
