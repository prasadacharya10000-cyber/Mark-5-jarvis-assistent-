"""
train_intent_model.py — Train / retrain the ML intent classifier for Mark XXXIX

Usage:
    python ml/train_intent_model.py
    python ml/train_intent_model.py --dataset ml/intents_dataset.json

To add new intents: edit ml/intents_dataset.json, add a new entry under "intents",
then re-run this script.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE_DIR     = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "intents_dataset.json"
MODEL_PATH   = BASE_DIR / "intent_model.pkl"


def load_dataset(path: Path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    X, y = [], []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            X.append(pattern.lower().strip())
            y.append(intent["tag"])
    return X, y, data


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10_000,
            sublinear_tf=True,
            strip_accents="unicode",
        )),
        ("clf", CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=2000)
        )),
    ])


def train(dataset_path: Path = DATASET_PATH, model_path: Path = MODEL_PATH):
    print(f"📂 Loading dataset: {dataset_path}")
    X, y, raw = load_dataset(dataset_path)

    num_intents = len(raw["intents"])
    print(f"✅ {len(X)} samples | {num_intents} intents")

    # Cross-validation
    pipeline = build_pipeline()
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"📊 CV Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")

    # Train/test split for detailed report
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Final fit on all data
    pipeline.fit(X, y)

    # Save
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"💾 Model saved to {model_path}")

    # Quick sanity checks
    test_phrases = [
        "open spotify",
        "what is the weather in london",
        "send a message to mom",
        "play lofi music on youtube",
        "set a reminder for 8am",
        "write a python scraper",
        "goodbye jarvis",
        "find flights to paris",
        "update my steam games",
        "take a screenshot",
    ]
    print("\n🤖 Quick Test:")
    for phrase in test_phrases:
        proba   = pipeline.predict_proba([phrase])[0]
        classes = pipeline.classes_
        top_idx = proba.argmax()
        print(f"  '{phrase}' → {classes[top_idx]} ({proba[top_idx]:.0%})")

    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Mark XXXIX intent classifier")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output",  type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    train(args.dataset, args.output)
