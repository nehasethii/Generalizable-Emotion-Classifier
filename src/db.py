import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ---------------------------
# Paths
# ---------------------------
MODEL_DIR = "models/DistilBERT"
VAL_CSV = "data/processed/goemotions_val_clean.csv"

# ---------------------------
# Load tokenizer and model
# ---------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# ---------------------------
# Load validation CSV
# ---------------------------
df = pd.read_csv(VAL_CSV)

texts = df["clean_text"].tolist()
labels = df["emotion"].tolist()

# ---------------------------
# Convert labels → IDs
# ---------------------------
label2id = model.config.label2id
id2label = model.config.id2label

y_true = [label2id[lbl] for lbl in labels]

# ---------------------------
# Tokenize all texts
# ---------------------------
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt"
)

# ---------------------------
# Predict
# ---------------------------
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    y_pred = torch.argmax(logits, dim=1).cpu().numpy()

# ---------------------------
# Metrics
# ---------------------------
accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted", zero_division=0
)

print("\n===== DistilBERT Evaluation =====")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)

print("\n===== Full Classification Report =====")
print(classification_report(
    y_true,
    y_pred,
    target_names=[id2label[i] for i in range(len(id2label))],
    zero_division=0
))
