# 🧠 Generalizable Emotion Classifier

A multi-model NLP system for fine-grained emotion classification that identifies **27 emotions + Neutral** from text using Machine Learning, Deep Learning, and Transformer-based architectures.

The project evaluates and compares multiple approaches including **Multinomial Naive Bayes (MNB)**, **LSTM**, **CNN-SVM Hybrid**, and **DistilBERT**, with additional support for explainable AI techniques and rule-based emotion overrides.

---

## 🚀 Features

- Detects **27 emotions + Neutral**
- Supports multiple classification models:
  - DistilBERT
  - LSTM
  - CNN + SVM Hybrid
  - Multinomial Naive Bayes
- Rule-based emotion override engine
- Explainable AI using LIME and SHAP
- Flask REST API backend
- Interactive Web Interface
- Batch CSV emotion prediction
- Comparative model evaluation framework

---

## 📊 Dataset

**GoEmotions Dataset (Google)**

- 58,009 annotated text samples
- 27 fine-grained emotions + Neutral
- Real-world conversational text
- Includes slang, abbreviations, sarcasm, and informal language

### Dataset Split

| Split | Samples |
|---------|---------|
| Training | 46,407 |
| Validation | 5,801 |
| Testing | 5,801 |
| Total | 58,009 |

---

## 🏗️ System Architecture

```text
Text Input
    ↓
Preprocessing
    ↓
Feature Extraction
    ↓
Rule-Based Override Engine
    ↓
Emotion Classification
    ↓
Interpretability Layer
    ↓
Web/API Deployment
```

---

## 🔍 Models Implemented

### 1. Multinomial Naive Bayes (MNB)

Classical baseline model using TF-IDF features.

- TF-IDF Vectorization
- N-gram Features
- Lightweight and Fast

### 2. LSTM

Deep learning sequence model.

- GloVe Embeddings
- Long-Term Dependency Learning
- Context-Aware Emotion Detection

### 3. CNN + SVM Hybrid

Hybrid architecture combining:

- CNN Feature Extraction
- Linear SVM Classification
- Softmax Probability Fallback

### 4. DistilBERT

Transformer-based model and final deployment choice.

- Contextual Embeddings
- Self-Attention Mechanism
- Transfer Learning
- Fine-tuned on GoEmotions Dataset

---

## 📈 Experimental Results

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score |
|---------|---------|---------|---------|---------|
| DistilBERT | **0.3272** | **0.2993** | **0.3272** | **0.2986** |
| MNB | 0.2538 | 0.2582 | 0.2538 | 0.1477 |
| LSTM | 0.1583 | 0.2133 | 0.1583 | 0.1449 |
| CNN + SVM | 0.1149 | 0.1195 | 0.1149 | 0.0902 |

### 🏆 Best Performing Model

**DistilBERT**

Why it performed best:

- Better contextual understanding
- Captures long-range semantic dependencies
- Strong generalization on unseen text

---

## 🔬 Explainable AI

To improve transparency and trust, the system integrates:

### LIME

Highlights important words contributing to predictions.

### SHAP

Provides token-level feature importance and model explanations.

This allows users to understand *why* a particular emotion was predicted.

---

## 🛠️ Tech Stack

### Languages

- Python
- JavaScript
- HTML
- CSS

### Machine Learning & NLP

- TensorFlow
- Keras
- PyTorch
- Scikit-Learn
- HuggingFace Transformers
- NLTK
- SpaCy

### Explainability

- LIME
- SHAP

### Backend

- Flask
- REST APIs

### Database & Utilities

- Pandas
- NumPy

---

## 📂 Project Structure

```text
Generalizable-Emotion-Classifier/
│
├── Backend/
├── Frontend/
├── data/
├── models/
│   ├── DistilBERT/
│   ├── lstm/
│   ├── cnn_svm/
│   └── mnb/
│
├── results/
├── src/
├── app.py
└── README.md
```

---

## 🎯 Future Improvements

- Multi-label emotion prediction
- Multilingual emotion classification
- Real-time chatbot integration
- Quantized transformer models

---

## 📄 Project Report

A detailed report covering methodology, experiments, model comparisons, and results is available here:

📑 [Project Report](Generalizable_Emotion_Classifier_Report.pdf)

## 👩‍💻 Author

**Neha Sethi**

B.Tech Information Technology  
Machine Learning & NLP Enthusiast  
Focused on AI, Problem Solving, and Software Development

---

> “Understanding human emotions through language is one step toward building more intelligent and empathetic AI systems.”
