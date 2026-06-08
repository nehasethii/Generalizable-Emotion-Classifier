# scripts/debug_predictors.py
import os, joblib, json
from datetime import datetime
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd()))
MNB_DIR = os.path.join(PROJECT_ROOT, "models", "mnb")
CNN_DIR = os.path.join(PROJECT_ROOT, "models", "cnn_svm")
TESTS = [
    "I am so happy and excited!",
    "This is terrible and makes me angry",
    "I feel sad and depressed"
]

print("MNB candidates in", MNB_DIR)
if os.path.isdir(MNB_DIR):
    for p in sorted(os.listdir(MNB_DIR)):
        if p.lower().endswith((".pkl", ".joblib")):
            full = os.path.join(MNB_DIR, p)
            print(" -", p, os.path.getsize(full), "bytes")
            try:
                obj = joblib.load(full)
                t = type(obj).__name__
                vs = None
                if hasattr(obj, "vocabulary_"):
                    vs = len(obj.vocabulary_)
                else:
                    try:
                        vs = len(obj.get_feature_names_out())
                    except Exception:
                        vs = None
                print("   type:", t, "vocab_size:", vs)
            except Exception as e:
                print("   load error:", e)

print("\nCNN-SVM candidates in", CNN_DIR)
if os.path.isdir(CNN_DIR):
    for p in sorted(os.listdir(CNN_DIR)):
        if p.lower().endswith((".pkl", ".joblib", ".h5")):
            full = os.path.join(CNN_DIR, p)
            print(" -", p, os.path.getsize(full), "bytes")

# Try import app's predictors if project structured as package
print("\nAttempting to import backend predictors (best-effort)...")
try:
    import sys
    sys.path.insert(0, PROJECT_ROOT)
    from backend import app as backend_app
    print("Imported backend.app")
    for s in TESTS:
        print("\nTest:", s)
        try:
            print(" MNB ->", backend_app.predict_mnb_text(s))
        except Exception as e:
            print(" MNB error:", e)
        try:
            print(" CNN ->", backend_app.predict_cnn_text(s))
        except Exception as e:
            print(" CNN error:", e)
except Exception as e:
    print("Could not import backend.app:", e)
    print("Run the server and make an API call to /api/predict to see runtime errors.")
