import pandas as pd
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

# ======================
# 📂 Paths
# ======================
PROCESSED_DIR = "data/processed"
MODELS_DIR = "models/mnb"
RESULTS_DIR = "results/mnb"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ======================
# ⚙️ Hyperparameters
# ======================
NB_ALPHA = 1.0
NB_FIT_PRIOR = True

VECT_MAX_FEATURES = 15000
VECT_NGRAM_RANGE = (1, 1)
VECT_MIN_DF = 3
VECT_MAX_DF = 0.85

# ======================
# 🔢 Manual Run Number
# ======================
# 👉 Change this each time you experiment
run_number = 1  # example: change to 4, 5, etc. for each new experiment

# ======================
# 📥 Load Data
# ======================
train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "goemotions_train_clean.csv"))
val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "goemotions_val_clean.csv"))

# Drop NaN rows if any remain
train_df.dropna(subset=["clean_text", "emotion"], inplace=True)
val_df.dropna(subset=["clean_text", "emotion"], inplace=True)

# ======================
# 🔠 Encode Labels
# ======================
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train_df["emotion"])
y_val = label_encoder.transform(val_df["emotion"])

# ======================
# ✨ Vectorize Text
# ======================
vectorizer = TfidfVectorizer(
    max_features=VECT_MAX_FEATURES,
    ngram_range=VECT_NGRAM_RANGE,
    min_df=VECT_MIN_DF,
    max_df=VECT_MAX_DF
)

X_train = vectorizer.fit_transform(train_df["clean_text"])
X_val = vectorizer.transform(val_df["clean_text"])

# ======================
# 🧠 Train Model
# ======================
model = MultinomialNB(alpha=NB_ALPHA, fit_prior=NB_FIT_PRIOR)
model.fit(X_train, y_train)

# ======================
# 🔍 Evaluate
# ======================
y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_val, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_val, y_pred, average="weighted", zero_division=0)

report = classification_report(
    y_val, y_pred,
    target_names=label_encoder.classes_,
    zero_division=0
)

print("\n📊 Detailed Classification Report:")
print(report)
print("\n✅ Overall Performance Summary:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

# ======================
# 💾 Save Model, Vectorizer, Encoder
# ======================
joblib.dump(model, os.path.join(MODELS_DIR, f"mnb_model_run{run_number}.pkl"))
joblib.dump(vectorizer, os.path.join(MODELS_DIR, f"tfidf_vectorizer_run{run_number}.pkl"))
joblib.dump(label_encoder, os.path.join(MODELS_DIR, f"label_encoder_run{run_number}.pkl"))

print("\n💾 Model and vectorizer saved successfully!")

# ======================
# 📄 Save Metrics + Confusion Matrix
# ======================
results_path = os.path.join(RESULTS_DIR, f"metrics_run{run_number}.txt")

with open(results_path, "w", encoding="utf-8") as f:
    f.write("=== Multinomial Naive Bayes Results ===\n\n")
    f.write("🔧 Hyperparameters:\n")
    f.write(f"  alpha       : {NB_ALPHA}\n")
    f.write(f"  fit_prior   : {NB_FIT_PRIOR}\n\n")
    f.write("🧮 Vectorizer Settings:\n")
    f.write(f"  max_features: {VECT_MAX_FEATURES}\n")
    f.write(f"  ngram_range : {VECT_NGRAM_RANGE}\n")
    f.write(f"  min_df      : {VECT_MIN_DF}\n")
    f.write(f"  max_df      : {VECT_MAX_DF}\n\n")
    f.write("--- Evaluation Metrics ---\n")
    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1-Score : {f1:.4f}\n\n")
    f.write("--- Classification Report ---\n")
    f.write(report)
    f.write(f"\nRun Time: {datetime.now()}\n")

print(f"📄 Results saved to {results_path}")

# ======================
# 📊 Confusion Matrix
# ======================
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=False,
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)
plt.title(f"Confusion Matrix - Run {run_number}")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

cm_path = os.path.join(RESULTS_DIR, f"confusion_run{run_number}.png")
plt.savefig(cm_path, dpi=300)
plt.close()

print(f"📊 Confusion matrix saved to {cm_path}")
