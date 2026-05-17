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
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("omw-1.4", quiet=True)

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
MODEL_PATH = "model.pkl"
METRICS_PATH = "model_metrics.pkl"

_STOP_BASE = set(stopwords.words("english"))
_KEEP = {"not", "no", "never", "nor", "neither", "barely", "hardly"}
STOP_WORDS = _STOP_BASE - _KEEP
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
    df = df[df["sentiment"].isin({"positive", "negative", "neutral"})].copy()
    return df


def train(df):
    df = df.copy()
    df["clean"] = df["review"].astype(str).apply(clean_text)
    df = df[df["clean"].str.strip() != ""].copy()

    X = list(df["clean"])
    y = list(df["sentiment"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    vec = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_df=0.95,
    )
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    clf = LogisticRegression(
        C=1.0,
        max_iter=500,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(Xtr, y_train)

    y_pred = clf.predict(Xte)
    y_prob = clf.predict_proba(Xte)
    classes = list(clf.classes_)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm   = confusion_matrix(y_test, y_pred, labels=classes)
    rep  = classification_report(y_test, y_pred, output_dict=True)

    Xfull = vec.transform(X)
    cv    = cross_val_score(clf, Xfull, y, cv=3, scoring="accuracy")

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
        "classes": classes,
        "report": rep,
        "cv_mean": float(cv.mean()),
        "cv_std": float(cv.std()),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    return vec, clf, metrics


def save_artifacts(vec, clf, metrics):
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"vectorizer": vec, "model": clf}, f)
        with open(METRICS_PATH, "wb") as f:
            pickle.dump(metrics, f)
    except Exception:
        pass


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
    vec, clf, metrics = train(df)
    save_artifacts(vec, clf, metrics)
    return vec, clf, metrics


def predict_single(text, vectorizer, model):
    cleaned = clean_text(str(text))
    features = vectorizer.transform([cleaned])
    label = model.predict(features)[0]
    proba = model.predict_proba(features)[0]
    classes = model.classes_
    prob_dict = {c: float(p) for c, p in zip(classes, proba)}
    return {
        "sentiment": label,
        "confidence": float(max(proba)),
        "probabilities": prob_dict,
    }


def predict_batch(texts, vectorizer, model):
    return [predict_single(t, vectorizer, model) for t in texts]


if __name__ == "__main__":
    print("Training...")
    df = load_dataset()
    vec, clf, metrics = train(df)
    save_artifacts(vec, clf, metrics)
    print("Accuracy:", round(metrics["accuracy"] * 100, 2), "%")
