import os
import json
import numpy as np

def _try_import(name):
    try:
        return __import__(name)
    except Exception:
        return None

def explain_prediction(model_name, text, loaded_models, model_dir):
    model_name = (model_name or "").lower()
    if model_name == "mnb":
        return explain_mnb(text, loaded_models.get("mnb"))
    if model_name in ("cnn_svm", "cnn-svm"):
        return explain_cnn_svm(text, loaded_models.get("cnn_svm"))
    if model_name == "lstm":
        return explain_lstm(text, loaded_models.get("lstm"), model_dir)
    if model_name in ("distilbert", "distil"):
        return explain_distilbert(text, loaded_models.get("distilbert"), model_dir)
    raise ValueError(f"Unknown model: {model_name}")


def _load_lime():
    # return LimeTextExplainer or None
    try:
        mod = _try_import("lime.lime_text")
        if mod:
            return mod.lime_text.LimeTextExplainer
    except Exception:
        pass
    try:
        from lime.lime_text import LimeTextExplainer  # type: ignore
        return LimeTextExplainer
    except Exception:
        return None

def _fallback_simple_vectorizer(text, vec):
    feats = []
    try:
        vocab = getattr(vec, "vocabulary_", None) or {}
        arr = vec.transform([text]).toarray()[0]
        idx = np.argsort(-np.abs(arr))[:8]
        inv = {v: k for k, v in vocab.items()}
        for i in idx:
            feats.append([inv.get(i, str(i)), float(arr[i])])
    except Exception:
        feats = [[text, 0.0]]
    return {"model": "mnb", "features": feats, "note": "LIME not installed; fallback used."}

def explain_mnb(text, bundle):
    """
    LIME explanation for MultinomialNB. Returns sentences with percentages.
    """
    LimeTextExplainer = _load_lime()
    if not bundle or "model" not in bundle or "vectorizer" not in bundle:
        raise ValueError("MNB bundle incomplete")

    clf = bundle["model"]
    vec = bundle["vectorizer"]
    le = bundle.get("label_encoder")
    class_names = le.classes_.tolist() if le is not None else None

    def predict_proba(texts):
        X = vec.transform(texts)
        try:
            return clf.predict_proba(X)
        except Exception:
            logits = clf.decision_function(X)
            return softmax_np(logits)

    if LimeTextExplainer is None:
        return _fallback_simple_vectorizer(text, vec)

    explainer = LimeTextExplainer(class_names=class_names)
    try:
        exp = explainer.explain_instance(text, predict_proba, num_features=8, top_labels=1)
    except Exception as e:
        return {"model": "mnb", "features": [], "error": str(e)}

    lbl = exp.available_labels()[0]
    feats = exp.as_list(label=lbl)
    top_label = class_names[lbl] if class_names else int(lbl)

    # convert to percentage of absolute weights
    weights = [abs(float(w)) for _, w in feats]
    total = sum(weights) + 1e-12
    sentences = [
        f'Word "{tok}" contributed { (abs(float(w)) / total) * 100:.1f}% to the prediction.'
        for tok, w in feats
    ]

    return {"model": "mnb", "top_label": top_label, "features": feats, "sentences": sentences}

def explain_cnn_svm(text, bundle):
    LimeTextExplainer = _load_lime()
    if not bundle or "cnn" not in bundle or "svm" not in bundle or "tokenizer" not in bundle:
        raise ValueError("cnn_svm bundle incomplete")

    cnn = bundle["cnn"]
    svm = bundle["svm"]
    tok = bundle["tokenizer"]

    # local import
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    def predict_proba(texts):
        seqs = tok.texts_to_sequences(texts)
        X = pad_sequences(seqs, maxlen=100)
        emb = cnn.predict(X)
        try:
            return svm.predict_proba(emb)
        except Exception:
            logits = svm.decision_function(emb)
            return softmax_np(logits)

    if LimeTextExplainer is None:
        return {"model": "cnn_svm", "features": [], "note": "LIME not installed."}

    explainer = LimeTextExplainer()
    exp = explainer.explain_instance(text, predict_proba, num_features=8, top_labels=1)
    lbl = exp.available_labels()[0]
    feats = exp.as_list(label=lbl)

    weights = [abs(float(w)) for _, w in feats]
    total = sum(weights) + 1e-12
    sentences = [
        f'Word "{tok}" contributed { (abs(float(w)) / total) * 100:.1f}% to the prediction.'
        for tok, w in feats
    ]

    return {"model": "cnn_svm", "top_label": int(lbl), "features": feats, "sentences": sentences}

def explain_lstm(text, lstm_model, model_dir):
    LimeTextExplainer = _load_lime()
    import pickle
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    if lstm_model is None:
        raise ValueError("LSTM model missing")

    lstm_dir = os.path.join(model_dir, "lstm")
    toks = [f for f in os.listdir(lstm_dir) if "tokenizer" in f.lower()]
    if not toks:
        raise ValueError("LSTM tokenizer missing")
    tokenizer_path = os.path.join(lstm_dir, toks[0])
    with open(tokenizer_path, "rb") as fh:
        tokenizer = pickle.load(fh)

    def predict_proba(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        X = pad_sequences(seqs, maxlen=100)
        return lstm_model.predict(X)

    if LimeTextExplainer is None:
        return {"model": "lstm", "features": [], "note": "LIME not installed."}

    explainer = LimeTextExplainer()
    exp = explainer.explain_instance(text, predict_proba, num_features=8, top_labels=1)
    lbl = exp.available_labels()[0]
    feats = exp.as_list(label=lbl)

    weights = [abs(float(w)) for _, w in feats]
    total = sum(weights) + 1e-12
    sentences = [
        f'Word "{tok}" contributed { (abs(float(w)) / total) * 100:.1f}% to the prediction.'
        for tok, w in feats
    ]

    return {"model": "lstm", "top_label": int(lbl), "features": feats, "sentences": sentences}

# =================================================
# ===============  SHAP - DistilBERT  =============
# =================================================
def explain_distilbert(text, bundle, model_dir):
    """
    SHAP-based explanation for DistilBERT (or HF model folder).
    Returns:
      - features: list of {token, value}
      - sentences: [ 'Word "tok" contributed X% to the prediction.' ]
      - top_label_index, top_label_prob
      - note when fallback or errors occur
    (Option A: Sentences-only; no HTML returned)
    """
    # dynamic imports
    transformers = _try_import("transformers")
    shap = _try_import("shap")
    torch = _try_import("torch")

    # check core libs, return friendly messages if missing
    if transformers is None:
        try:
            import transformers  # type: ignore
            transformers = transformers
        except Exception as e:
            return {"model": "distilbert", "note": f"transformers not available: {e}"}
    if shap is None:
        try:
            import shap  # type: ignore
            shap = shap
        except Exception as e:
            return {"model": "distilbert", "note": f"shap not available: {e}"}

    # Import needed HF classes
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

    # Prefer provided bundle tokenizer/model; otherwise try to load from model_dir
    tokenizer = None
    model = None
    if bundle and isinstance(bundle, dict):
        tokenizer = bundle.get("tokenizer")
        model = bundle.get("model")

    # Try candidate paths
    candidate_paths = []
    if model_dir:
        candidate_paths.append(os.path.join(model_dir, "DistilBERT"))
        candidate_paths.append(os.path.join(model_dir, "models", "DistilBERT"))
        candidate_paths.append(model_dir)
    model_path = next((p for p in candidate_paths if p and os.path.exists(p)), None)

    if tokenizer is None or model is None:
        if model_path is None:
            return {"model": "distilbert", "note": f"DistilBERT folder not found under model_dir; provide bundle or place model under 'DistilBERT'."}
    try:
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_path or "distilbert-base-uncased", local_files_only=True)
        if model is None:
            model = AutoModelForSequenceClassification.from_pretrained(model_path or "distilbert-base-uncased", local_files_only=True)
    except Exception as e:
        return {"model": "distilbert", "note": f"Failed to load tokenizer/model from {model_path or 'default'}: {e}"}

    # Build pipeline where available (use CPU if no GPU)
    nlp_pipeline = None
    try:
        device = 0 if (torch is not None and torch.cuda.is_available()) else -1
        nlp_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True, device=device)
    except Exception:
        nlp_pipeline = None

    # Helper: ensure batch is list[str]
    def _coerce_batch(x):
        if x is None:
            return []
        if isinstance(x, str):
            return [x]
        if isinstance(x, (list, tuple, np.ndarray)):
            return [str(i) for i in x]
        return [str(x)]

    # Prediction wrapper expected by SHAP: accepts list[str] (or single str) and returns np.ndarray (n_samples, n_classes)
    def pred_fn(inputs):
        batch = _coerce_batch(inputs)
        if len(batch) == 0:
            nlab = getattr(model.config, "num_labels", 1)
            return np.zeros((0, nlab), dtype=float)

        # If HF pipeline available, use it (returns list[list[dict]] or list[dict])
        if nlp_pipeline is not None:
            outs = nlp_pipeline(batch)
            probs = []
            for item in outs:
                if isinstance(item, dict):
                    # fallback: convert dict to list of scores
                    scores = []
                    id2label = getattr(model.config, "id2label", None)
                    if isinstance(id2label, dict):
                        try:
                            ordered = [id2label[i] for i in sorted(id2label.keys(), key=lambda k: int(k) if str(k).isdigit() else k)]
                            for lab in ordered:
                                # pipeline dict may use label keys or 'score' mapping; try to be robust
                                scores.append(float(item.get("score", item.get(lab, 0.0)) if isinstance(item.get("score", None), (int, float)) else item.get(lab, 0.0)))
                        except Exception:
                            scores = [float(v) for v in item.values()]
                    else:
                        scores = [float(item.get("score", 0.0))]
                    probs.append(scores)
                elif isinstance(item, (list, tuple)):
                    vals = [float(d.get("score", 0.0)) for d in item]
                    probs.append(vals)
                else:
                    try:
                        probs.append([float(item)])
                    except Exception:
                        probs.append([0.0])
            arr = np.array(probs, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape((arr.shape[0], 1))
            return arr

        # Manual tokenization + model forward (PyTorch expected)
        enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
        try:
            if torch is not None and hasattr(model, "parameters"):
                device = next(model.parameters()).device
                enc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}
        except Exception:
            pass

        if torch is not None:
            ctx = torch.no_grad()
        else:
            class _DummyCtx:
                def __enter__(self): return None
                def __exit__(self, exc_type, exc, tb): return False
            ctx = _DummyCtx()

        with ctx:
            try:
                out = model(**enc)
            except Exception as e:
                raise RuntimeError(f"Model forward failed: {e}")

        logits = getattr(out, "logits", out)
        if hasattr(logits, "detach"):
            logits = logits.detach()
        if hasattr(logits, "cpu"):
            logits_np = logits.cpu().numpy()
        else:
            logits_np = np.asarray(logits)

        try:
            import scipy.special
            probs = scipy.special.softmax(logits_np, axis=-1)
        except Exception:
            arr = np.array(logits_np, dtype=float)
            if arr.ndim == 1:
                e = np.exp(arr - np.max(arr))
                probs = (e / np.sum(e)).reshape(1, -1)
            else:
                e = np.exp(arr - np.max(arr, axis=1, keepdims=True))
                probs = e / np.sum(e, axis=1, keepdims=True)

        probs = np.asarray(probs, dtype=float)
        if probs.ndim == 1:
            probs = probs.reshape((probs.shape[0], 1))
        return probs

    # Prepare batch_texts as list[str] (SHAP expects that)
    batch_texts = _coerce_batch(text)
    if len(batch_texts) == 0:
        return {"model": "distilbert", "note": "Empty input text."}

    # Build masker and explainer
    try:
        masker = shap.maskers.Text(tokenizer)
    except Exception as e:
        return {"model": "distilbert", "note": f"Failed to create SHAP masker: {e}"}

    try:
        explainer = shap.Explainer(pred_fn, masker)
    except Exception as e:
        return {"model": "distilbert", "note": f"Failed to construct SHAP explainer: {e}"}

    # Run SHAP
    try:
        shap_values = explainer(batch_texts)
    except Exception as e:
        return {"model": "distilbert", "note": f"SHAP failed: {e}"}

    # Extract tokens and values and turn into sentences
    try:
        probs = pred_fn(batch_texts)[0]
        top_idx = int(np.argmax(probs)) if probs.size else 0

        vals = shap_values.values
        toks = shap_values.data[0]  # token pieces for first example

        # Normalize shapes to token-level vector
        if isinstance(vals, np.ndarray):
            if vals.ndim == 3:
                token_vals = vals[0][:, top_idx]
            elif vals.ndim == 2:
                token_vals = vals[0]
            elif vals.ndim == 1:
                token_vals = vals
            else:
                token_vals = np.asarray(vals).flatten()
        else:
            token_vals = np.asarray(vals).flatten()

        # Merge subword pieces into readable words (handles common markers: ##, ▁, Ġ)
        merged_tokens = []
        merged_vals = []
        def _append(tok_piece, v):
            merged_tokens.append(tok_piece)
            merged_vals.append(float(v))

        for piece, v in zip(toks, token_vals):
            s = str(piece)
            # BERT-style wordpieces "##" (attach to previous)
            if s.startswith("##"):
                if merged_tokens:
                    merged_tokens[-1] = merged_tokens[-1] + s[2:]
                    merged_vals[-1] = merged_vals[-1] + float(v)
                else:
                    _append(s[2:], v)
            # SentencePiece / other tokenizers often prefix with "▁" or "Ġ"
            elif s.startswith("▁") or s.startswith("Ġ"):
                cleaned = s[1:]
                _append(cleaned, v)
            else:
                _append(s, v)

        # If nothing merged, fall back to original tokens
        if not merged_tokens:
            merged_tokens = [str(x) for x in toks]
            merged_vals = list(map(float, np.asarray(token_vals).flatten()))

        # ---------------------------
        # FILTER OUT EMPTY / SPECIAL TOKENS
        # ---------------------------
        # Basic blacklist of common special tokens
        specials = {"", "[CLS]", "[SEP]", "[PAD]", "[UNK]", "<s>", "</s>", "[CLS]", "[SEP]"}
        # include tokenizer special tokens where available
        try:
            if tokenizer is not None:
                for attr in ("cls_token", "sep_token", "pad_token", "unk_token", "bos_token", "eos_token"):
                    tokval = getattr(tokenizer, attr, None)
                    if tokval:
                        specials.add(str(tokval))
        except Exception:
            pass

        filtered_tokens = []
        filtered_vals = []
        for tok, v in zip(merged_tokens, merged_vals):
            # strip whitespace and common invisible characters
            if tok is None:
                continue
            cleaned = str(tok).strip()
            # skip empty / whitespace-only tokens
            if cleaned == "":
                continue
            # skip tokens that are only punctuation or are in specials
            if cleaned in specials:
                continue
            # skip tokens that are just punctuation (e.g. ".", ",") or single-control chars
            if all(ch in ".,;:!?\"'()[]{}-—…" for ch in cleaned):
                continue
            # passed filters -> keep (use trimmed token)
            filtered_tokens.append(cleaned)
            filtered_vals.append(float(v))

        # If filtering removed everything, fall back to merged lists (but still strip empties)
        if not filtered_tokens:
            for tok, v in zip(merged_tokens, merged_vals):
                cleaned = (str(tok) if tok is not None else "").strip()
                if cleaned == "":
                    continue
                filtered_tokens.append(cleaned)
                filtered_vals.append(float(v))

        # Convert absolute contributions into percentages
        abs_vals = [abs(v) for v in filtered_vals]
        total = sum(abs_vals) + 1e-12
        # Build sentences like other models: Word "..." contributed XX.X% to the prediction.
        sentences = []
        features = []
        for tok, v in zip(filtered_tokens, filtered_vals):
            pct = (abs(float(v)) / total) * 100.0 if total > 0 else 0.0
            sentences.append(f'Word "{tok}" contributed {pct:.1f}% to the prediction.')
            features.append({"token": tok, "value": float(v)})

        return {
            "model": "distilbert",
            "top_label_index": top_idx,
            "top_label_prob": float(probs[top_idx]) if len(probs) > 0 else None,
            "features": features,
            "sentences": sentences,
            "note": "SHAP token-level explanation (merged subwords; empty/special tokens filtered)."
        }

    except Exception as e:
        return {"model": "distilbert", "note": f"Failed to extract SHAP tokens: {e}"}

# =================================================
# =============== Utilities =======================
# =================================================
def softmax_np(arr):
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        e = np.exp(a - np.max(a))
        return e / np.sum(e)
    if a.ndim == 2:
        e = np.exp(a - np.max(a, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)
    flat = a.flatten()
    e = np.exp(flat - np.max(flat))
    s = e / np.sum(e)
    return s.reshape(a.shape)
