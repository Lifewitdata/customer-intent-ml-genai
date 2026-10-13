"""Tests for model training and inference."""

import numpy as np

from src.intent.data import load_raw
from src.intent.models import predict_baseline, train_baseline


def test_baseline_train_evaluate(tmp_path):
    train, test = load_raw()
    # Use a small slice for speed.
    train = train.groupby("label").head(5).reset_index(drop=True)
    test = test.groupby("label").head(2).reset_index(drop=True)
    clf, vectorizer, le, metrics = train_baseline(train, test)
    assert metrics["accuracy"] >= 0.0
    assert metrics["macro_f1"] >= 0.0
    assert len(le.classes_) > 0


def test_predict_baseline_shape_and_fields():
    texts = ["my card is expiring", "i was charged twice"]
    preds = predict_baseline(texts)
    assert len(preds) == len(texts)
    for p in preds:
        assert set(p.keys()) == {"intent", "confidence", "model"}
        assert isinstance(p["intent"], str)
        assert 0 <= p["confidence"] <= 1
        assert p["model"] == "baseline"
