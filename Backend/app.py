# backend/app.py  (FULL - drop-in replacement)
import os
import sys
import fnmatch
import logging
import joblib
import numpy as np
import re
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# PROJECT ROOT
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Paths
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MNB_DIR = os.path.join(MODELS_DIR, "mnb")
CNN_DIR = os.path.join(MODELS_DIR, "cnn_svm")
LSTM_DIR = os.path.join(MODELS_DIR, "lstm")

DISTIL_DIR = None
for cand in ("DistilBERT", "distilbert", "distilbert_emotion"):
    p = os.path.join(MODELS_DIR, cand)
    if os.path.isdir(p):
        DISTIL_DIR = p
        break

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "predictions")
os.makedirs(RESULTS_DIR, exist_ok=True)

# canonical emotions (GoEmotions 27)
CANONICAL_EMOTIONS = [
    'admiration','amusement','anger','annoyance','approval','caring','confusion','curiosity',
    'desire','disappointment','disapproval','disgust','embarrassment','excitement','fear',
    'gratitude','grief','joy','love','nervousness','optimism','pride','realization','relief',
    'remorse','sadness','surprise'
]

# Flask app
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Serve index at root
@app.route("/", methods=["GET"])
def serve_index():
    try:
        return app.send_static_file("home.html")
    except Exception:
        return "Index not found", 404

# -------------------------
# Rule-based overrides (simple regex rules)
# -------------------------
RULES = [
    (re.compile(r"\b(admirable|respect|admire|great job|well done|impressive)\b", re.I), "admiration"),
    (re.compile(r"\b(lol|haha|rofl|funny|hilarious)\b", re.I), "amusement"),
    (re.compile(r"\b(angry|furious|rage|mad|pissed|annoyed)\b", re.I), "anger"),
    (re.compile(r"\b(annoyed|ugh|frustrat)\b", re.I), "annoyance"),
    (re.compile(r"\b(agree|well said|approved)\b", re.I), "approval"),
    (re.compile(r"\b(care|concerned|take care)\b", re.I), "caring"),
    (re.compile(r"\b(confusing|confused|dont understand|what the)\b", re.I), "confusion"),
    (re.compile(r"\b(curio|curious|wondering|want to know)\b", re.I), "curiosity"),
    (re.compile(r"\b(want|wish|desire|craving)\b", re.I), "desire"),
    (re.compile(r"\b(disappoint|let down|underwhelming)\b", re.I), "disappointment"),
    (re.compile(r"\b(disapprove|not okay|wrong)\b", re.I), "disapproval"),
    (re.compile(r"\b(disgust|gross|eww|nasty)\b", re.I), "disgust"),
    (re.compile(r"\b(embarrass|awkward|cringe)\b", re.I), "embarrassment"),
    (re.compile(r"\b(excited|thrill|cant wait|pumped)\b", re.I), "excitement"),
    (re.compile(r"\b(scared|afraid|fear|terrified|panic)\b", re.I), "fear"),
    (re.compile(r"\b(thank you|thanks|grateful|appreciate)\b", re.I), "gratitude"),
    (re.compile(r"\b(grief|mourning|condolence|rip)\b", re.I), "grief"),
    (re.compile(r"\b(joy|happy|glad|delighted|smiling)\b", re.I), "joy"),
    (re.compile(r"\b(love|adore|in love|heart)\b", re.I), "love"),
    (re.compile(r"\b(nervous|anxious|worried|uneasy)\b", re.I), "nervousness"),
    (re.compile(r"\b(hopeful|optimistic|positive)\b", re.I), "optimism"),
    (re.compile(r"\b(proud|accomplish|achievement)\b", re.I), "pride"),
    (re.compile(r"\b(realiz|now I see|oh now)\b", re.I), "realization"),
    (re.compile(r"\b(relieved|phew|thank god|finally)\b", re.I), "relief"),
    (re.compile(r"\b(regret|sorry|my fault)\b", re.I), "remorse"),
    (re.compile(r"\b(sad|cry|depress|upset|heartbroken)\b", re.I), "sadness"),
    (re.compile(r"\b(wow|surpris|unexpected|shocked|no way)\b", re.I), "surprise"),
]

def rule_override(text: str):
    for pat, lab in RULES:
        if pat.search(text):
            return lab
    return None

# -------------------------
# Utilities
# -------------------------
def find_file(directory, patterns):
    if not directory or not os.path.isdir(directory):
        return None
    for root, _, files in os.walk(directory):
        for pat in patterns:
            for f in fnmatch.filter(files, pat):
                return os.path.join(root, f)
    return None

def softmax(x):
    arr = np.array(x, dtype=float)
    if arr.ndim == 1:
        ex = np.exp(arr - np.max(arr))
        return ex / np.sum(ex)
    if arr.ndim == 2:
        ex = np.exp(arr - np.max(arr, axis=1, keepdims=True))
        return ex / np.sum(ex, axis=1, keepdims=True)
    arr_flat = arr.flatten()
    ex = np.exp(arr_flat - np.max(arr_flat))
    sm = ex / np.sum(ex)
    return sm.reshape(arr.shape)

def ensure_1d(arr):
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape[0] == 1:
        return a[0]
    return a.reshape(-1)

# -------------------------
# Load models (best-effort)
# -------------------------
log.info("Loading models ...")

# LSTM
lstm = None
lstm_tok = None
lstm_le = None
try:
    from tensorflow.keras.models import load_model
    import pickle
    h5 = find_file(LSTM_DIR, ['lstm_model*.h5', 'lstm*.h5', '*.h5'])
    if h5:
        lstm = load_model(h5)
        log.info("LSTM loaded: %s", h5)
    tpath = find_file(LSTM_DIR, ['tokenizer*.pkl', 'tokenizer.pkl'])
    if tpath:
        with open(tpath, 'rb') as fh:
            lstm_tok = pickle.load(fh)
    lepath = find_file(LSTM_DIR, ['label_encoder*.pkl', 'label_encoder.pkl'])
    if lepath:
        lstm_le = joblib.load(lepath)
except Exception:
    log.exception("LSTM load failed or tensorflow missing")

# MNB
mnb = None
tfidf = None
mnb_le = None
try:
    mnb_p = find_file(MNB_DIR, ['mnb_model*.pkl', 'multinomial_nb.pkl', 'mnb_model.pkl'])
    if mnb_p:
        mnb = joblib.load(mnb_p)
        log.info("MNB loaded: %s", mnb_p)
    tfidf_p = find_file(MNB_DIR, ['tfidf_vectorizer*.pkl', 'tfidf_vectorizer*.joblib'])
    if tfidf_p:
        tfidf = joblib.load(tfidf_p)
        log.info("TF-IDF loaded: %s", tfidf_p)
    lep = find_file(MNB_DIR, ['label_encoder*.pkl', 'label_encoder.pkl'])
    if lep:
        mnb_le = joblib.load(lep)
except Exception:
    log.exception("MNB load failed")

# CNN + SVM
cnn_enc = None
cnn_tok = None
svm = None
cnn_le = None
try:
    enc_p = find_file(CNN_DIR, ['cnn_encoder*.h5', '*encoder*.h5', '*.h5'])
    if enc_p:
        from tensorflow.keras.models import load_model as _load_model
        cnn_enc = _load_model(enc_p)
        log.info("CNN encoder loaded: %s", enc_p)
    tokj = find_file(CNN_DIR, ['tokenizer.json'])
    if tokj:
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
        with open(tokj, 'r', encoding='utf-8') as fh:
            cnn_tok = tokenizer_from_json(fh.read())
    svmp = find_file(CNN_DIR, ['*svm*.joblib', '*svm*.pkl', '*.joblib', '*.pkl'])
    if svmp:
        svm = joblib.load(svmp)
        log.info("SVM loaded: %s", svmp)
    lep = find_file(CNN_DIR, ['label_encoder*.joblib', 'label_encoder.joblib', 'label_encoder.pkl'])
    if lep:
        cnn_le = joblib.load(lep)
except Exception:
    log.exception("CNN+SVM load failed")

# DistilBERT (optional)
distil_tokenizer = None
distil_model = None
distil_id2label = None
try:
    if DISTIL_DIR:
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        import torch
        distil_tokenizer = DistilBertTokenizerFast.from_pretrained(DISTIL_DIR, local_files_only=True)
        distil_model = DistilBertForSequenceClassification.from_pretrained(DISTIL_DIR, local_files_only=True)
        try:
            distil_id2label = distil_model.config.id2label
        except Exception:
            distil_id2label = None
        log.info("DistilBERT loaded.")
except Exception:
    log.exception("DistilBERT load failed (transformers/torch may be missing)")

log.info("Load summary: LSTM=%s MNB=%s TFIDF=%s CNN_enc=%s SVM=%s Distil=%s",
         bool(lstm), bool(mnb), bool(tfidf), bool(cnn_enc), bool(svm), bool(distil_model))

# -------------------------
# Helpers to produce canonical labels + sorted top list
# -------------------------
def get_labels_from_encoder(le):
    if le is None:
        return CANONICAL_EMOTIONS[:]  # fallback
    try:
        return list(le.classes_)
    except Exception:
        try:
            return list(le)
        except Exception:
            return CANONICAL_EMOTIONS[:]

def build_output_from_probs(probs_array, labels_list=None):
    """
    Returns dict with:
      - labels: list of label strings in same order as probs
      - probs: list of floats
      - top: list of dicts sorted descending by prob [{label, index, prob},...]
    """
    probs = ensure_1d(probs_array).astype(float)
    if labels_list is None:
        labels = CANONICAL_EMOTIONS[:len(probs)]
    else:
        labels = list(labels_list)
        # if mismatch, try to adapt
        if len(labels) != len(probs):
            if len(labels) > len(probs):
                labels = labels[:len(probs)]
            elif len(probs) == len(CANONICAL_EMOTIONS):
                labels = CANONICAL_EMOTIONS[:]
            else:
                # fallback numeric labels
                labels = [str(i) for i in range(len(probs))]

    idxs = np.argsort(probs)[::-1]  # descending
    top = []
    for i in idxs:
        top.append({"label": labels[int(i)], "index": int(i), "prob": float(probs[int(i)])})
    return {"labels": labels, "probs": probs.tolist(), "top": top}

# -------------------------
# Prediction helper functions (return organized dicts)
# -------------------------
def predict_lstm(text):
    if not lstm or not lstm_tok:
        return {"error": "LSTM or its tokenizer not loaded"}
    seq = lstm_tok.texts_to_sequences([text])
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    X = pad_sequences(seq, maxlen=100)
    probs = lstm.predict(X)
    probs = ensure_1d(probs)
    labels = get_labels_from_encoder(lstm_le)
    out = build_output_from_probs(probs, labels_list=labels)
    pred_idx = int(np.argmax(probs)) if probs.size > 0 else None
    pred_label = labels[pred_idx] if pred_idx is not None and pred_idx < len(labels) else (int(pred_idx) if pred_idx is not None else None)
    out.update({"model": "lstm", "prediction": pred_label})
    return out

def predict_mnb(text):
    if not mnb or not tfidf:
        return {"error": "MNB or TF-IDF vectorizer not loaded"}
    X = tfidf.transform([text])
    try:
        probs = mnb.predict_proba(X)
        probs = ensure_1d(probs)
    except Exception:
        logits = mnb.decision_function(X)
        logits = ensure_1d(logits)
        probs = softmax(logits)
    labels = get_labels_from_encoder(mnb_le)
    out = build_output_from_probs(probs, labels_list=labels)
    pred_idx = int(np.argmax(probs)) if probs.size > 0 else None
    pred_label = labels[pred_idx] if pred_idx is not None and pred_idx < len(labels) else (int(pred_idx) if pred_idx is not None else None)
    out.update({"model": "mnb", "prediction": pred_label})
    return out

def predict_cnn_svm(text):
    if not cnn_enc or not cnn_tok or not svm:
        return {"error": "CNN encoder / tokenizer / SVM not loaded"}
    seq = cnn_tok.texts_to_sequences([text])
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    X = pad_sequences(seq, maxlen=100)
    emb = cnn_enc.predict(X)
    emb = np.asarray(emb)
    probs = None
    if hasattr(svm, "predict_proba"):
        try:
            probs = svm.predict_proba(emb)
            probs = ensure_1d(probs)
        except Exception:
            probs = None
    if probs is None:
        try:
            logits = svm.decision_function(emb)
            logits = ensure_1d(logits)
            probs = softmax(logits)
        except Exception:
            try:
                pred = svm.predict(emb)
                classes = getattr(svm, "classes_", np.unique(pred))
                ncls = len(classes)
                probs = np.zeros(ncls)
                probs[int(pred[0])] = 1.0
            except Exception as e:
                return {"error": "SVM prediction failed", "detail": str(e)}
    labels = get_labels_from_encoder(cnn_le)
    out = build_output_from_probs(probs, labels_list=labels)
    pred_idx = int(np.argmax(probs)) if probs.size > 0 else None
    pred_label = labels[pred_idx] if pred_idx is not None and pred_idx < len(labels) else (int(pred_idx) if pred_idx is not None else None)
    out.update({"model": "cnn_svm", "prediction": pred_label})
    return out

def predict_distil(text):
    if not distil_model or not distil_tokenizer:
        return {"error": "DistilBERT not loaded"}
    import torch
    toks = distil_tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        out = distil_model(**toks)
    logits = out.logits.cpu().numpy()
    logits = ensure_1d(logits)
    probs = softmax(logits)

    # Build labels list from distil_id2label when present
    labels = None
    if isinstance(distil_id2label, dict):
        # distil_id2label might be like {0: 'joy', 1:'sadness'} or {'LABEL_0':'joy',...}
        try:
            # prefer numeric-keyed dict
            numeric_keys = [k for k in distil_id2label.keys() if str(k).isdigit()]
            if numeric_keys:
                # sort by numeric keys
                keys_sorted = sorted(numeric_keys, key=lambda k: int(k))
                labels = [distil_id2label[k] for k in keys_sorted]
            else:
                # fallback: if keys are 'LABEL_0' ... sort by suffix number
                try:
                    keys_sorted = sorted(distil_id2label.keys(), key=lambda k: int(re.sub(r'\D', '', str(k)) or 0))
                    labels = [distil_id2label[k] for k in keys_sorted]
                except Exception:
                    labels = list(distil_id2label.values())
        except Exception:
            labels = list(distil_id2label.values())
    else:
        # if no mapping provided, fallback to canonical but trimmed/padded
        labels = CANONICAL_EMOTIONS[:len(probs)]

    # If length mismatches, adapt:
    if len(labels) != len(probs):
        if len(labels) > len(probs):
            labels = labels[:len(probs)]
        elif len(probs) == len(CANONICAL_EMOTIONS):
            labels = CANONICAL_EMOTIONS[:]
        else:
            labels = [str(i) for i in range(len(probs))]

    # Build output and ensure top is sorted descending (build_output_from_probs already sorts)
    out = build_output_from_probs(probs, labels_list=labels)
    pred_idx = int(np.argmax(probs)) if probs.size > 0 else None
    pred_label = labels[pred_idx] if pred_idx is not None and pred_idx < len(labels) else (int(pred_idx) if pred_idx is not None else None)
    out.update({"model": "distilbert", "prediction": pred_label})
    return out

# -------------------------
# API endpoints
# -------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.json or {}
    text = body.get("text", "") or ""
    model = (body.get("model", "distilbert") or "distilbert").lower()

    # Apply rule override first
    r = rule_override(text)
    if r:
        return jsonify({"model": "rule_override", "prediction": r, "probs": None, "labels": CANONICAL_EMOTIONS, "top": [{"label": r, "prob": 1.0}]})

    # Dispatch
    if model == "lstm":
        return jsonify(predict_lstm(text))
    if model == "mnb":
        return jsonify(predict_mnb(text))
    if model == "cnn_svm" or model == "cnn-svm":
        return jsonify(predict_cnn_svm(text))
    if model == "distilbert" or model == "distil":
        return jsonify(predict_distil(text))
    return jsonify({"error": "unknown model"}), 400

# Interpretability - uses interpret.py helper if present
try:
    from interpret import explain_prediction
except Exception:
    explain_prediction = None
    log.exception("Could not import interpret.py - interpret endpoint will be unavailable")

@app.route("/api/interpret/<model>", methods=["POST"])
def api_interpret(model):
    if explain_prediction is None:
        return jsonify({"error": "interpret module not available"}), 500
    text = (request.json or {}).get("text", "")
    try:
        expl = explain_prediction(model, text,
            loaded_models={
                "mnb": {"model": mnb, "vectorizer": tfidf, "label_encoder": mnb_le},
                "lstm": lstm,
                "cnn_svm": {"cnn": cnn_enc, "svm": svm, "tokenizer": cnn_tok, "label_encoder": cnn_le},
                "distilbert_paths": DISTIL_DIR
            },
            model_dir=MODELS_DIR
        )
        return jsonify({"model": model, "explanation": expl})
    except Exception as e:
        log.exception("Interpret error")
        return jsonify({"error": str(e)}), 500

# Debug endpoints
@app.route("/api/debug/files")
def debug_files():
    return jsonify({
        "mnb_loaded": bool(mnb),
        "tfidf_loaded": bool(tfidf),
        "cnn_encoder_loaded": bool(cnn_enc),
        "cnn_svm_loaded": bool(svm),
        "lstm_loaded": bool(lstm),
        "distil_loaded": bool(distil_model)
    })

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now().isoformat()})

# Run server
if __name__ == "__main__":
    log.info("Server started at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000)
