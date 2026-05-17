"""
train_model.py  -  Sentiment Analysis ML Pipeline
Compatible with Python 3.8+ and Streamlit Cloud.
"""

import os
import re
import pickle
import string
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("omw-1.4",   quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

DATASET_PATH = "dataset.csv"
MODEL_PATH   = "model.pkl"
METRICS_PATH = "model_metrics.pkl"

_STOP_WORDS_BASE = set(stopwords.words("english"))
_KEEP = {"not", "no", "never", "nor", "neither", "barely", "hardly"}
STOP_WORDS = _STOP_WORDS_BASE - _KEEP
LEMMATIZER = WordNetLemmatizer()


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|<.*?>", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b\d+\b", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS or t in _KEEP]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


def load_dataset(path=DATASET_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError("dataset.csv not found.")
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    df.dropna(subset=["review", "sentiment"], inplace=True)
    df["sentiment"] = df["sentiment"].str.strip().str.lower()
    valid = {"positive", "negative", "neutral"}
    df = df[df["sentiment"].isin(valid)].copy()
    return df


def train(df):
    df = df.copy()
    df["clean_review"] = df["review"].astype(str).apply(clean_text)
    df = df[df["clean_review"].str.strip() != ""].copy()

    X = np.array(df["clean_review"].tolist())
    y = np.array(df["sentiment"].tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_df=0.90,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    model = LogisticRegression(
        C=1.5,
        max_iter=1000,
        solver="lbfgs",
        multi_class="multinomial",
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train_tfidf, y_train)

    y_pred  = model.predict(X_test_tfidf)
    y_prob  = model.predict_proba(X_test_tfidf)
    classes = list(model.classes_)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred, labels=classes)
    rep  = classification_report(y_test, y_pred, output_dict=True)

    X_full = vectorizer.transform(X)
    cv     = cross_val_score(model, X_full, y, cv=5, scoring="accuracy")

    metrics = {
        "accuracy"        : acc,
        "precision"       : prec,
        "recall"          : rec,
        "f1"              : f1,
        "confusion_matrix": cm,
        "classes"         : classes,
        "report"          : rep,
        "cv_mean"         : float(cv.mean()),
        "cv_std"          : float(cv.std()),
        "X_test"          : X_test,
        "y_test"          : y_test,
        "y_pred"          : y_pred,
        "y_prob"          : y_prob,
    }
    return vectorizer, model, metrics


def save_artifacts(vectorizer, model, metrics):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "model": model}, f)
    with open(METRICS_PATH, "wb") as f:
        pickle.dump(metrics, f)


def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    metrics = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "rb") as f:
            metrics = pickle.load(f)
    return bundle["vectorizer"], bundle["model"], metrics


def get_model():
    if os.path.exists(MODEL_PATH):
        try:
            return load_artifacts()
        except Exception:
            pass
    df = load_dataset()
    vectorizer, model, metrics = train(df)
    try:
        save_artifacts(vectorizer, model, metrics)
    except Exception:
        pass
    return vectorizer, model, metrics


def predict_single(text, vectorizer, model):
    cleaned  = clean_text(str(text))
    features = vectorizer.transform([cleaned])
    label    = model.predict(features)[0]
    proba    = model.predict_proba(features)[0]
    classes  = model.classes_
    prob_dict = {c: float(p) for c, p in zip(classes, proba)}
    return {
        "sentiment"    : label,
        "confidence"   : float(max(proba)),
        "probabilities": prob_dict,
    }


def predict_batch(texts, vectorizer, model):
    return [predict_single(t, vectorizer, model) for t in texts]


if __name__ == "__main__":
    print("Loading dataset...")
    df = load_dataset()
    print("Training model...")
    vectorizer, model, metrics = train(df)
    print("Saving artifacts...")
    save_artifacts(vectorizer, model, metrics)
    print("Done! Accuracy:", round(metrics["accuracy"] * 100, 2), "%")
