"""Feature extraction for the classical baseline."""

from __future__ import annotations

import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "models"


def build_vectorizer(max_features: int = 20_000, ngram_range=(1, 2)) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
    )


def save_vectorizer(vectorizer: TfidfVectorizer, path: Path | None = None) -> None:
    path = path or MODELS_DIR / "tfidf_vectorizer.joblib"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)


def load_vectorizer(path: Path | None = None) -> TfidfVectorizer:
    path = path or MODELS_DIR / "tfidf_vectorizer.joblib"
    return joblib.load(path)
