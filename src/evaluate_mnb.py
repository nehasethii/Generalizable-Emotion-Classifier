# src/evaluate_mnb.py
import os, joblib, json
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from src.rules import apply_rules

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(ROOT, "models", "mnb")
TEST_CSV = os.path.join(ROOT, "data", "processed", "goemotions_test_clean.csv")
RESULTS_DIR = os.path.join(ROOT, "results", "mnb")
os.makedirs(RESULTS_DIR, exist_ok=True)

def find_file(directory, patterns):
    import fnmatch
    for root, _, files in os.walk(directory):
        for p in patterns:
            for f in fnmatch.filter(files, p):
                return os.path.join(root, f)
    return None

def load():
    tfidf_p = find_file(MODEL_DIR, ["tfidf_vectorizer*.pkl","tfidf_vectorizer.pkl"])
    clf_p = find_file(MODEL_DIR, ["mnb_model*.pkl","multinomial_nb.pkl","mnb_model.pkl"])
    le_p = find_file(MODEL_DIR, ["label_encoder*.pkl","label_encoder.pkl"])
    if not tfidf_p or not clf_p or not le_p:
        raise FileNotFoundError("MNB artifacts not found in " + MODEL_DIR)
    tfidf = joblib.load(tfidf_p)
    clf = joblib.load(clf_p)
    le = joblib.load(le_p)
    return tfidf, clf, le

def evaluate():
    tfidf, clf, le = load()
    df = pd.read_csv(TEST_CSV)
    df = df.dropna(subset=["clean_text","emotion"])
    texts = df["clean_text"].astype(str).tolist()
    true = df["emotion"].tolist()

    # rule overrides applied first
    preds = []
    for t in texts:
        rlabel, rule = apply_rules(t)
        if rlabel:
            preds.append(rlabel)
            continue
        X = tfidf.transform([t])
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X)[0]
            idx = int(np.argmax(probs))
            plabel = le.inverse_transform([idx])[0]
        else:
            plabel = clf.predict(X)[0]
        preds.append(plabel)

    # metrics
    y_true = true
    y_pred = preds
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(le.classes_))

    out = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "report": report
    }

    # save results
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    df_out = df.copy()
    df_out["pred"] = y_pred
    df_out.to_csv(os.path.join(RESULTS_DIR, f"mnb_test_preds_{ts}.csv"), index=False, encoding="utf-8")
    with open(os.path.join(RESULTS_DIR, f"mnb_metrics_{ts}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("MNB metrics:", out)
    return out

if __name__ == "__main__":
    evaluate()
