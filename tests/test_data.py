"""Tests for data loading."""

import pandas as pd

from src.intent.data import load_label_map, load_raw


def test_load_raw_returns_dataframes():
    train, test = load_raw()
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert set(train.columns) == {"text", "label"}
    assert set(test.columns) == {"text", "label"}
    assert len(train) > len(test)


def test_labels_are_strings():
    train, _ = load_raw()
    assert pd.api.types.is_string_dtype(train["label"])
    assert all(isinstance(x, str) for x in train["label"].head(10))


def test_label_map_exists_and_covers_intents():
    train, _ = load_raw()
    label_map = load_label_map()
    assert isinstance(label_map, list)
    assert len(label_map) == 77
    assert set(train["label"].unique()).issubset(set(label_map))
