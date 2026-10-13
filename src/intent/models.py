"""Classical and transformer intent classifiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.intent.data import load_label_map, load_raw
from src.intent.evaluate import print_report, report
from src.intent.features import build_vectorizer, save_vectorizer

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"


def train_baseline(train, test, max_iter: int = 1000, C: float = 10.0):
    """Train a TF-IDF + logistic regression baseline."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Banking77's test split contains all 77 intents but train covers only 70, so the encoder must
    # be fit on the shared label space or le.transform(test) raises on unseen intents.
    le = LabelEncoder()
    le.fit(pd.concat([train["label"], test["label"]]).unique())
    y_train = le.transform(train["label"])
    y_test = le.transform(test["label"])

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(train["text"])
    X_test = vectorizer.transform(test["text"])

    clf = LogisticRegression(max_iter=max_iter, C=C, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    metrics = report(y_test, y_pred, label_names=le.classes_)
    print("Baseline model metrics:")
    print_report(metrics)

    joblib.dump(clf, MODELS_DIR / "baseline_clf.joblib")
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    save_vectorizer(vectorizer)
    (MODELS_DIR / "baseline_metrics.json").write_text(
        json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2),
        encoding="utf-8",
    )
    return clf, vectorizer, le, metrics


def train_transformer(
    train,
    test,
    model_name: str = "distilbert-base-uncased",
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    tiny: bool = False,
):
    """Fine-tune a transformer for intent classification."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = MODELS_DIR / "transformer"

    le = LabelEncoder()
    le.fit(train["label"])
    num_labels = len(le.classes_)

    if tiny:
        model_name = "prajjwal1/bert-tiny"
        epochs = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=64)

    train_ds = Dataset.from_pandas(train[["text", "label"]])
    test_ds = Dataset.from_pandas(test[["text", "label"]])
    train_ds = train_ds.map(lambda row: {"label": le.transform([row["label"]])[0]})
    test_ds = test_ds.map(lambda row: {"label": le.transform([row["label"]])[0]})
    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    args = TrainingArguments(
        output_dir=str(out_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        seed=42,
    )

    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds)
    trainer.train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    joblib.dump(le, MODELS_DIR / "transformer_label_encoder.joblib")
    return model, tokenizer, le


def load_baseline():
    clf = joblib.load(MODELS_DIR / "baseline_clf.joblib")
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    le = joblib.load(MODELS_DIR / "label_encoder.joblib")
    return clf, vectorizer, le


def load_transformer():
    out_dir = MODELS_DIR / "transformer"
    model = AutoModelForSequenceClassification.from_pretrained(out_dir)
    tokenizer = AutoTokenizer.from_pretrained(out_dir)
    le = joblib.load(MODELS_DIR / "transformer_label_encoder.joblib")
    return model, tokenizer, le


def predict_baseline(texts: list[str]) -> list[dict]:
    clf, vectorizer, le = load_baseline()
    X = vectorizer.transform(texts)
    probs = clf.predict_proba(X)
    top = np.argmax(probs, axis=1)
    return [
        {"intent": le.classes_[i], "confidence": float(probs[j, i]), "model": "baseline"}
        for j, i in enumerate(top)
    ]


def predict_transformer(texts: list[str]) -> list[dict]:
    model, tokenizer, le = load_transformer()
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).numpy()
    top = np.argmax(probs, axis=1)
    return [
        {"intent": le.classes_[i], "confidence": float(probs[j, i]), "model": "distilbert"}
        for j, i in enumerate(top)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train both models")
    parser.add_argument("--tiny", action="store_true", help="Use tiny BERT for quick tests")
    args = parser.parse_args()

    if args.train:
        train, test = load_raw()
        print("Training baseline...")
        train_baseline(train, test)
        print("\nTraining transformer...")
        train_transformer(train, test, tiny=args.tiny)


if __name__ == "__main__":
    main()
