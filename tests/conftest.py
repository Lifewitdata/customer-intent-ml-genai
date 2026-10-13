"""Shared test fixtures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "raw"


@pytest.fixture(scope="session", autouse=True)
def ensure_artifacts():
    """Create minimal baseline artifacts so API and agent tests can run without full training."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    texts = [
        "my card is expiring",
        "when will my card arrive",
        "i was charged twice",
        "i lost my card",
        "how do i change my pin",
    ]
    labels = [
        "card_about_to_expire",
        "card_delivery_estimate",
        "transaction_charged_twice",
        "lost_or_stolen_card",
        "change_pin",
    ]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    joblib.dump(clf, MODELS_DIR / "baseline_clf.joblib")
    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")

    label_map_path = DATA_DIR / "label_map.json"
    if not label_map_path.exists():
        label_map_path.write_text(json.dumps(le.classes_.tolist()), encoding="utf-8")

    yield

    # Optional cleanup: remove test artifacts.
    # shutil.rmtree(MODELS_DIR, ignore_errors=True)
