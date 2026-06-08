# src/train_cnn_svm_run.py
"""
Train CNN (supervised) then extract features and train SVM — single configurable run.
Edit hyperparameters at the top (LEARNING_RATE set to 0.001 as requested).
Saves encoder (.h5), svm (.joblib) and appends a result row to results/cnn_svm/cnn_svm_grid_results.csv
Logs progress to console with timestamps.
"""

import os, joblib, gc, traceback, time
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.utils import class_weight

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# ---------------- USER HYPERPARAMETERS ----------------
PROCESSED_DIR = "data/processed"
MODEL_DIR = "models/cnn_svm"
RESULTS_DIR = "results/cnn_svm"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(RESULTS_DIR, "cnn_svm_grid_results.csv")

# Change these as needed:
MAX_WORDS = 30000
MAX_LEN = 100
EMBED_DIM = 100
FILTERS = 128

BATCH_SIZE = 32
LEARNING_RATE = 1e-4   
EPOCHS = 10            
SVM_KERNEL = "rbf"
SVM_C = 5.0
BATCH_PRED = 128
# ----------------------------------------------------

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"[{now()}] START single CNN+SVM run | lr={LEARNING_RATE} bs={BATCH_SIZE} epochs={EPOCHS}")

# ------------- Load data -------------
print(f"[{now()}] Loading processed CSVs from {PROCESSED_DIR}")
train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "goemotions_train_clean.csv"))
val_df   = pd.read_csv(os.path.join(PROCESSED_DIR, "goemotions_val_clean.csv"))
train_df.dropna(subset=["clean_text","emotion"], inplace=True)
val_df.dropna(subset=["clean_text","emotion"], inplace=True)
texts_train = train_df["clean_text"].astype(str).tolist()
texts_val   = val_df["clean_text"].astype(str).tolist()

# ------------- Label encoder -------------
le = LabelEncoder()
y_train = le.fit_transform(train_df["emotion"])
y_val   = le.transform(val_df["emotion"])
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
print(f"[{now()}] Label encoder saved")

# ------------- Tokenizer -------------
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(texts_train)
with open(os.path.join(MODEL_DIR, "tokenizer.json"), "w") as f:
    f.write(tokenizer.to_json())
print(f"[{now()}] Tokenizer saved")

X_train = pad_sequences(tokenizer.texts_to_sequences(texts_train), maxlen=MAX_LEN)
X_val   = pad_sequences(tokenizer.texts_to_sequences(texts_val),   maxlen=MAX_LEN)
print(f"[{now()}] Data shapes -> X_train: {X_train.shape}, X_val: {X_val.shape}")

# ------------- Class weights -------------
cw = class_weight.compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
cw_dict = dict(enumerate(cw))
print(f"[{now()}] Computed class weights")

# ------------- Model builder -------------
def build_cnn_classifier(num_classes, filters=FILTERS, embed_dim=EMBED_DIM, max_words=MAX_WORDS, max_len=MAX_LEN, lr=LEARNING_RATE):
    inp = Input(shape=(max_len,))
    emb = Embedding(input_dim=max_words, output_dim=embed_dim, input_length=max_len)(inp)
    c1 = Conv1D(filters, 3, activation="relu")(emb)
    c2 = Conv1D(filters, 4, activation="relu")(emb)
    c3 = Conv1D(filters, 5, activation="relu")(emb)
    p1 = GlobalMaxPooling1D()(c1)
    p2 = GlobalMaxPooling1D()(c2)
    p3 = GlobalMaxPooling1D()(c3)
    concat = Concatenate()([p1, p2, p3])
    x = Dense(256, activation="relu")(concat)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ------------- Train CNN (supervised) -------------
try:
    tf.keras.backend.clear_session()
    num_classes = len(le.classes_)
    model = build_cnn_classifier(num_classes=num_classes)
    print(f"[{now()}] Starting CNN training...")
    t0 = time.time()
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE, class_weight=cw_dict, verbose=1)
    t1 = time.time()
    print(f"[{now()}] CNN training finished (elapsed {t1-t0:.1f}s)")
except Exception:
    traceback.print_exc()
    raise

# ------------- Create encoder (penultimate Dense=256) -------------
# robust search for Dense(256)
encoder_output = None
for layer in model.layers[::-1]:
    if getattr(layer, "units", None) == 256:
        encoder_output = layer.output
        break
if encoder_output is None:
    # fallback: pick layer before final Dense
    encoder_output = model.layers[-2].output
encoder = Model(inputs=model.input, outputs=encoder_output)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
encoder_path = os.path.join(MODEL_DIR, f"cnn_encoder_ep{EPOCHS}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_{stamp}.h5")
encoder.save(encoder_path)
print(f"[{now()}] Encoder saved at: {encoder_path}")

# ------------- Feature extraction -------------
print(f"[{now()}] Starting feature extraction (train)...")
t0 = time.time()
feats_train = encoder.predict(X_train, batch_size=BATCH_PRED, verbose=1)
t1 = time.time()
print(f"[{now()}] Finished train features (elapsed {t1-t0:.1f}s)")

print(f"[{now()}] Starting feature extraction (val)...")
t0 = time.time()
feats_val = encoder.predict(X_val, batch_size=BATCH_PRED, verbose=1)
t1 = time.time()
print(f"[{now()}] Finished val features (elapsed {t1-t0:.1f}s)")

# ------------- Train SVM (with verbose & fallback) -------------
svm_path = os.path.join(MODEL_DIR, f"svm_ep{EPOCHS}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_{stamp}.joblib")
try:
    print(f"[{now()}] Training SVC (RBF, verbose). This will print progress in console.")
    from sklearn.svm import SVC
    svm = SVC(kernel=SVM_KERNEL, probability=True, C=SVM_C, gamma="scale", verbose=True)
    t0 = time.time()
    svm.fit(feats_train, y_train)
    t1 = time.time()
    joblib.dump(svm, svm_path)
    print(f"[{now()}] SVC finished & saved (elapsed {t1-t0:.1f}s): {svm_path}")
except Exception as e:
    print(f"[{now()}] SVC failed or too slow: {e}. Falling back to SGD+CalibratedClassifierCV.")
    from sklearn.linear_model import SGDClassifier
    from sklearn.calibration import CalibratedClassifierCV
    linear = SGDClassifier(loss="log", max_iter=1000)
    cal = CalibratedClassifierCV(linear, cv=3)
    t0 = time.time()
    cal.fit(feats_train, y_train)
    t1 = time.time()
    joblib.dump(cal, svm_path)
    print(f"[{now()}] Fallback classifier saved (elapsed {t1-t0:.1f}s): {svm_path}")

# ------------- Evaluate & save results -------------
clf = joblib.load(svm_path)
probs = clf.predict_proba(feats_val)
preds = np.argmax(probs, axis=1)

acc = accuracy_score(y_val, preds)
prec = precision_score(y_val, preds, average="weighted", zero_division=0)
rec = recall_score(y_val, preds, average="weighted", zero_division=0)
f1 = f1_score(y_val, preds, average="weighted", zero_division=0)

res = {
    "run_time": datetime.now().isoformat(),
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "filters": FILTERS,
    "embedding_dim": EMBED_DIM,
    "svm_kernel": SVM_KERNEL,
    "svm_C": SVM_C,
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1_score": f1,
    "encoder_path": encoder_path,
    "svm_path": svm_path
}

# append to CSV
if os.path.exists(RESULTS_CSV):
    df_prev = pd.read_csv(RESULTS_CSV)
    df_out = pd.concat([df_prev, pd.DataFrame([res])], ignore_index=True)
else:
    df_out = pd.DataFrame([res])
df_out.to_csv(RESULTS_CSV, index=False)
print(f"[{now()}] Results appended to {RESULTS_CSV}")
print(f"[{now()}] DONE. acc={acc:.4f} f1={f1:.4f}")

# ------------- cleanup -------------
tf.keras.backend.clear_session()
gc.collect()
