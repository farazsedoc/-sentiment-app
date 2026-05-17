"""
train_model.py
==============
Trains the Sentiment Analysis ML model using TF-IDF + Logistic Regression.
Run this file once to generate the model.pkl and vectorizer.pkl files.

Usage:
    python train_model.py
"""

import os
import re
import pickle
import string
import warnings
import numpy as np
import pandas as pd

# ── NLTK Downloads ──────────────────────────────────────────────────────────
import nltk
# Suppress download messages for cleaner output
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Scikit-learn Imports ─────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# ── Constants ────────────────────────────────────────────────────────────────
DATASET_PATH = "dataset.csv"
MODEL_PATH   = "model.pkl"
METRICS_PATH = "model_metrics.pkl"

STOP_WORDS   = set(stopwords.words('english'))
LEMMATIZER   = WordNetLemmatizer()

# Words to KEEP even though they are stopwords (negations matter for sentiment)
SENTIMENT_NEGATIONS = {'not', 'no', 'never', 'nor', "n't", 'neither', 'barely', 'hardly'}
STOP_WORDS = STOP_WORDS - SENTIMENT_NEGATIONS


# ════════════════════════════════════════════════════════════════════════════
#  TEXT PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline for a raw review string.

    Steps:
        1. Lowercase conversion
        2. URL and HTML tag removal
        3. Punctuation removal
        4. Numeric removal
        5. Tokenization
        6. Stopword removal (preserving negations)
        7. Lemmatization
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 1 – lowercase
    text = text.lower()

    # Step 2 – remove URLs and HTML tags
    text = re.sub(r'http\S+|www\S+|<.*?>', ' ', text)

    # Step 3 – remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Step 4 – remove standalone numbers (keep word-embedded ones)
    text = re.sub(r'\b\d+\b', ' ', text)

    # Step 5 – tokenize
    tokens = text.split()

    # Step 6 – stopword removal (keep negation words)
    tokens = [t for t in tokens if t not in STOP_WORDS or t in SENTIMENT_NEGATIONS]

    # Step 7 – lemmatize
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    return ' '.join(tokens)


# ════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & VALIDATION
# ════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """Load and validate the dataset CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. "
            "Please ensure dataset.csv is in the same directory."
        )

    df = pd.read_csv(path)

    # Basic column validation
    required_cols = {'review', 'sentiment'}
    missing = required_cols - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()

    # Drop rows where review or sentiment is missing
    df.dropna(subset=['review', 'sentiment'], inplace=True)

    # Normalize sentiment labels
    df['sentiment'] = df['sentiment'].str.strip().str.lower()
    valid_labels = {'positive', 'negative', 'neutral'}
    df = df[df['sentiment'].isin(valid_labels)].copy()

    print(f"  Loaded {len(df)} valid records from '{path}'")
    print(f"  Label distribution:\n{df['sentiment'].value_counts().to_string()}\n")

    return df


# ════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ════════════════════════════════════════════════════════════════════════════

def train(df: pd.DataFrame):
    """
    Train a TF-IDF + Logistic Regression model.
    Returns the fitted vectorizer, model, and metric dict.
    """
    print("  Preprocessing reviews …")
    df['clean_review'] = df['review'].apply(clean_text)
    df = df[df['clean_review'].str.strip() != ''].copy()

    X = df['clean_review'].values
    y = df['sentiment'].values

    # ── Train / Test split (80 / 20, stratified) ─────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── TF-IDF Vectorizer ─────────────────────────────────────────────────
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),      # unigrams + bigrams
        sublinear_tf=True,       # apply log-normalization to TF
        min_df=2,                # ignore very rare terms
        max_df=0.90,             # ignore near-universal terms
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    # ── Logistic Regression Classifier ───────────────────────────────────
    model = LogisticRegression(
        C=1.5,
        max_iter=1000,
        solver='lbfgs',
        multi_class='multinomial',
        random_state=42,
        class_weight='balanced',
    )
    model.fit(X_train_tfidf, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────
    y_pred     = model.predict(X_test_tfidf)
    y_prob     = model.predict_proba(X_test_tfidf)
    classes    = model.classes_

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm        = confusion_matrix(y_test, y_pred, labels=classes)
    report    = classification_report(y_test, y_pred, output_dict=True)

    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print()

    # Cross-validation score on full training set
    X_full_tfidf = vectorizer.transform(X)
    cv_scores    = cross_val_score(model, X_full_tfidf, y, cv=5, scoring='accuracy')
    print(f"  5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

    metrics = {
        'accuracy'        : accuracy,
        'precision'       : precision,
        'recall'          : recall,
        'f1'              : f1,
        'confusion_matrix': cm,
        'classes'         : list(classes),
        'report'          : report,
        'cv_mean'         : cv_scores.mean(),
        'cv_std'          : cv_scores.std(),
        'X_test'          : X_test,
        'y_test'          : y_test,
        'y_pred'          : y_pred,
        'y_prob'          : y_prob,
    }

    return vectorizer, model, metrics


# ════════════════════════════════════════════════════════════════════════════
#  PERSISTENCE
# ════════════════════════════════════════════════════════════════════════════

def save_artifacts(vectorizer, model, metrics):
    """Serialize the vectorizer, model, and metrics to disk."""
    bundle = {
        'vectorizer': vectorizer,
        'model'     : model,
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(METRICS_PATH, 'wb') as f:
        pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Model  saved → {MODEL_PATH}")
    print(f"  Metrics saved → {METRICS_PATH}")


def load_artifacts():
    """Load and return (vectorizer, model, metrics) from disk."""
    with open(MODEL_PATH, 'rb') as f:
        bundle = pickle.load(f)

    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'rb') as f:
            metrics = pickle.load(f)

    return bundle['vectorizer'], bundle['model'], metrics


# ════════════════════════════════════════════════════════════════════════════
#  PUBLIC API  (used by app.py)
# ════════════════════════════════════════════════════════════════════════════

def get_model():
    """
    Return (vectorizer, model, metrics).
    Trains and saves the model on first call if no saved model is found.
    """
    if os.path.exists(MODEL_PATH):
        return load_artifacts()

    print("[SentimentML] No saved model found — training from scratch …\n")
    df = load_dataset()
    vectorizer, model, metrics = train(df)
    save_artifacts(vectorizer, model, metrics)
    return vectorizer, model, metrics


def predict_single(text: str, vectorizer, model) -> dict:
    """
    Predict sentiment for a single review string.

    Returns:
        {
            'sentiment': 'positive' | 'negative' | 'neutral',
            'confidence': float (0–1),
            'probabilities': {'positive': float, 'negative': float, 'neutral': float}
        }
    """
    cleaned   = clean_text(text)
    features  = vectorizer.transform([cleaned])
    label     = model.predict(features)[0]
    proba     = model.predict_proba(features)[0]
    classes   = model.classes_

    prob_dict = {c: float(p) for c, p in zip(classes, proba)}
    confidence = float(max(proba))

    return {
        'sentiment'    : label,
        'confidence'   : confidence,
        'probabilities': prob_dict,
    }


def predict_batch(texts: list, vectorizer, model) -> list:
    """
    Predict sentiment for a list of review strings.
    Returns a list of result dicts (same shape as predict_single).
    """
    results = []
    for text in texts:
        results.append(predict_single(text, vectorizer, model))
    return results


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print(" Sentiment Analysis Model Training")
    print("=" * 60)
    print()

    print("[1/3] Loading dataset …")
    df = load_dataset()

    print("[2/3] Training model …")
    vectorizer, model, metrics = train(df)

    print("[3/3] Saving artifacts …")
    save_artifacts(vectorizer, model, metrics)

    print()
    print("=" * 60)
    print(" Training complete. Run 'streamlit run app.py' to launch.")
    print("=" * 60)
