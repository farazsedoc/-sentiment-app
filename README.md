# 🧠 SentiScope — Sentiment Analysis Dashboard
### BCA Final Year Project | ML-Powered Product Review Analyzer

---

## 📁 Project Structure

```
sentiment_app/
├── app.py               ← Main Streamlit application (all pages + UI)
├── train_model.py       ← ML training pipeline (TF-IDF + Logistic Regression)
├── dataset.csv          ← Labelled review dataset (100 reviews)
├── model.pkl            ← Auto-generated: saved model & vectorizer
├── model_metrics.pkl    ← Auto-generated: evaluation metrics
├── requirements.txt     ← Python package dependencies
└── README.md            ← This file
```

---

## ⚡ Quick Start

### 1. Install Python (3.10 or later)
Download from https://python.org

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Pre-train the model
The app trains automatically on first run, but you can trigger it manually:
```bash
python train_model.py
```

### 5. Launch the dashboard
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 🔑 Demo Login Credentials

| Username | Password  |
|----------|-----------|
| admin    | admin123  |
| demo     | demo2024  |
| guest    | guest     |

---

## 🤖 ML Pipeline Summary

| Step | Component | Detail |
|------|-----------|--------|
| Preprocessing | NLTK | Lowercase, punctuation removal, stopword removal, lemmatization |
| Vectorization | TF-IDF | Unigrams + bigrams, 8,000 features, sublinear TF |
| Classifier | Logistic Regression | Multinomial, C=1.5, balanced class weights |
| Evaluation | Scikit-learn | Accuracy, Precision, Recall, F1, Confusion Matrix, 5-Fold CV |
| Persistence | Pickle | model.pkl + model_metrics.pkl |

---

## 📊 App Features

- **Dashboard** — KPI cards, sentiment trend, pie/bar/rating charts, word cloud, recent reviews
- **All Reviews** — Search, filter, sort, paginate; CSV batch upload with download
- **Submit Review** — Live preview prediction as you type
- **Analytics** — Model metrics, confusion matrix, classification report, per-class breakdown
- **Live Analyzer** — Instant prediction with confidence meter and keyword detection
- **About / Dictionary** — Project docs, ML pipeline, glossary, project structure

---

## 📦 Dataset Format

The CSV must have at minimum these two columns:

```
review,sentiment
"Great product!",positive
"Terrible quality.",negative
"It's okay.",neutral
```

Optional columns: `rating` (1–5), `date` (YYYY-MM-DD)

---

*Developed as BCA Final Year Project — Sentiment Analysis using Machine Learning, 2024*
