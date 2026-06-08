# src/evaluate_all.py
"""
Robust aggregator for per-model evaluation scripts.

Improvements over previous version:
- Better TF-IDF auto-selection logic for MNB: prefers exact vocab-size match to model.coef_ if available; otherwise selects vectorizer with smallest absolute vocab-size difference and, among ties, the most recently modified file.
- Chooses MNB model heuristically (prefers files with 'mnb' or 'multinomial' in name; otherwise the largest .pkl).
- Writes a detailed debug file with discovered candidate vectorizers and selection rationale.
- Keeps previous behavior of attempting each evaluator and saving combined CSV/plot.

Run from project root:
    python -m src.evaluate_all
"""
import os
import json
import glob
import joblib
import traceback
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_CSV = os.path.join(ROOT, "results", "overall_metrics.csv")
OUT_PLOT = os.path.join(ROOT, "results", "overall_model_comparison.png")
DEBUG_PATH = os.path.join(ROOT, "results", "evaluation_debug.json")
os.makedirs(os.path.dirname(OUT_PLOT), exist_ok=True)

from importlib import import_module

EVALS = [
    ("MNB", "src.evaluate_mnb", "evaluate"),
    ("LSTM", "src.evaluate_lstm", "evaluate"),
    ("CNN+SVM", "src.evaluate_cnn_svm", "evaluate"),
    ("DistilBERT", "src.evaluate_distilbert", "evaluate"),
]

results = []
errors = {}
debug = {"mnb_candidates": []}

# Helper functions for TF-IDF selection
import fnmatch

def find_files(directory, patterns):
    found = []
    if not directory or not os.path.isdir(directory):
        return found
    for root, _, files in os.walk(directory):
        for pat in patterns:
            for f in fnmatch.filter(files, pat):
                found.append(os.path.join(root, f))
    return found

def choose_mnb_model(mnb_dir):
    # prefer filenames containing 'mnb' or 'multinomial'; otherwise choose largest pkl
    cand = find_files(mnb_dir, ['*mnb*.pkl', '*multinomial*.pkl'])
    if cand:
        return cand[0]
    all_pkl = find_files(mnb_dir, ['*.pkl'])
    if not all_pkl:
        return None
    # choose largest file as fallback
    sizes = [(os.path.getsize(p), p) for p in all_pkl]
    sizes.sort(reverse=True)
    return sizes[0][1]

def inspect_vectorizer(path):
    try:
        v = joblib.load(path)
        if hasattr(v, 'vocabulary_'):
            vocab_size = len(v.vocabulary_)
        else:
            try:
                vocab_size = len(v.get_feature_names_out())
            except Exception:
                vocab_size = None
        mtime = os.path.getmtime(path)
        return {"path": path, "vocab_size": vocab_size, "mtime": mtime}
    except Exception as e:
        return {"path": path, "vocab_size": None, "mtime": None, "error": str(e)}

# Auto-fix logic improved
def try_autofix_and_eval_mnb():
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    from src.rules import apply_rules
    MNB_DIR = os.path.join(ROOT, "models", "mnb")
    TEST_CSV = os.path.join(ROOT, "data", "processed", "goemotions_test_clean.csv")

    if not os.path.isdir(MNB_DIR):
        raise FileNotFoundError("models/mnb directory not found")

    mnb_path = choose_mnb_model(MNB_DIR)
    if not mnb_path:
        raise FileNotFoundError("No candidate MNB model found in models/mnb")
    mnb = joblib.load(mnb_path)

    # expected feature width from coef_ if present
    expected = None
    try:
        expected = int(mnb.coef_.shape[1])
    except Exception:
        expected = None

    # discover candidate vectorizers
    cand_vecs = find_files(MNB_DIR, ['tfidf_*.pkl', 'tfidf_vectorizer*.pkl', '*.pkl'])
    inspected = []
    for p in cand_vecs:
        info = inspect_vectorizer(p)
        inspected.append(info)

    debug['mnb_candidates'] = inspected

    # filter only those with numeric vocab_size
    with_vocab = [c for c in inspected if c.get('vocab_size') is not None]

    chosen = None
    reason = None
    if expected is not None and with_vocab:
        # look for exact match
        exact = [c for c in with_vocab if c['vocab_size'] == expected]
        if exact:
            # if multiple, pick most recent
            exact.sort(key=lambda x: x['mtime'] or 0, reverse=True)
            chosen = exact[0]['path']
            reason = f"exact match vocab_size={expected}"
        else:
            # choose closest vocab_size by absolute diff, tiebreaker: most recent
            with_vocab.sort(key=lambda x: (abs(x['vocab_size'] - expected), -(x['mtime'] or 0)))
            chosen = with_vocab[0]['path']
            reason = f"closest vocab_size to expected={expected} (chosen {with_vocab[0]['vocab_size']})"
    else:
        # if expected unknown, choose latest vectorizer with a vocab size
        if with_vocab:
            with_vocab.sort(key=lambda x: x['mtime'] or 0, reverse=True)
            chosen = with_vocab[0]['path']
            reason = "latest available vectorizer"
        else:
            # no vectorizer with vocab info; attempt to pick largest pkl
            pkl_files = find_files(MNB_DIR, ['*.pkl'])
            if pkl_files:
                pkl_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                chosen = pkl_files[0]
                reason = "fallback: most recent pkl"

    debug['mnb_selection'] = {"model": mnb_path, "expected_features": expected, "chosen_vectorizer": chosen, "reason": reason}

    if not chosen:
        raise RuntimeError("Could not determine a TF-IDF vectorizer for MNB auto-fix")

    vec = joblib.load(chosen)

    # attempt to load label encoder
    le_candidates = find_files(MNB_DIR, ['label_encoder*.pkl', '*label*encoder*.pkl'])
    label_enc = None
    if le_candidates:
        try:
            label_enc = joblib.load(le_candidates[0])
            debug['mnb_label_encoder'] = le_candidates[0]
        except Exception as e:
            debug['mnb_label_encoder_error'] = str(e)

    df = pd.read_csv(TEST_CSV).dropna(subset=['clean_text','emotion'])
    texts = df['clean_text'].astype(str).tolist()
    y_true = df['emotion'].tolist()

    y_pred = []
    for t in texts:
        rl, _ = apply_rules(t)
        if rl:
            y_pred.append(rl)
            continue
        X = vec.transform([t])
        if hasattr(mnb, 'predict_proba'):
            probs = mnb.predict_proba(X)[0]
            idx = int(np.argmax(probs))
            if label_enc is not None:
                pl = label_enc.inverse_transform([idx])[0]
            else:
                try:
                    pl = mnb.classes_[idx]
                except Exception:
                    pl = str(idx)
        else:
            pl = mnb.predict(X)[0]
        y_pred.append(pl)

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

# Run evaluators robustly
for name, module_path, fn_name in EVALS:
    print('========== Running:', name)
    try:
        mod = import_module(module_path)
        fn = getattr(mod, fn_name)
        out = fn()
        results.append({'model': name, 'accuracy': out['accuracy'], 'precision': out['precision'], 'recall': out['recall'], 'f1': out['f1']})
    except Exception as e:
        tb = traceback.format_exc()
        print(f'Evaluator {name} failed:', e)
        errors[name] = str(e)
        # If MNB failed, try autofix
        if name == 'MNB':
            try:
                print('Attempting improved auto-fix for MNB...')
                out = try_autofix_and_eval_mnb()
                results.append({'model': name, 'accuracy': out['accuracy'], 'precision': out['precision'], 'recall': out['recall'], 'f1': out['f1']})
                print('Auto-fix MNB succeeded')
            except Exception as e2:
                print('Auto-fix for MNB failed:', e2)
                errors[name + '_autofix'] = str(e2)

# Save results and plot
if results:
    df = pd.DataFrame(results)
    if os.path.exists(OUT_CSV):
        prev = pd.read_csv(OUT_CSV)
        df_out = pd.concat([prev, df], ignore_index=True)
    else:
        df_out = df
    df_out.to_csv(OUT_CSV, index=False)

    ax = df.plot(x='model', y=['accuracy', 'f1'], kind='bar', figsize=(8,5), ylim=(0,1))
    ax.set_ylabel('Score')
    ax.set_title('Model comparison on test set')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=200)
    print('Saved overall metrics to:', OUT_CSV)
    print('Saved comparison plot to:', OUT_PLOT)
else:
    print('No successful results to save.')

# Save debug info and errors
debug['errors'] = errors
with open(DEBUG_PATH, 'w') as f:
    json.dump(debug, f, indent=2)
if errors:
    err_path = os.path.join(ROOT, 'results', 'evaluation_errors.json')
    with open(err_path, 'w') as f:
        json.dump(errors, f, indent=2)
    print('Some evaluators failed. See:', err_path)

print('Done.')
