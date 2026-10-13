"""Load and cache the Banking77 dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
LABEL_MAP_FILE = DATA_DIR / "label_map.json"


def load_raw(as_dataframe: bool = True):
    """Download Banking77 from Hugging Face and return train/test.

    Args:
        as_dataframe: if True, return (train_df, test_df) with string labels;
            otherwise return the raw `datasets.Dataset` objects.
    """
    ds = load_dataset("banking77")
    if not as_dataframe:
        return ds["train"], ds["test"]

    label_names = ds["train"].features["label"].names
    train = _to_frame(ds["train"], label_names)
    test = _to_frame(ds["test"], label_names)
    _save_label_map(label_names)
    return train, test


def _to_frame(split, label_names):
    return pd.DataFrame(
        {
            "text": [t.strip() for t in split["text"]],
            "label": [label_names[i] for i in split["label"]],
        }
    )


def _save_label_map(label_names):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_MAP_FILE.write_text(json.dumps(label_names, indent=2), encoding="utf-8")


def load_label_map() -> list[str]:
    return json.loads(LABEL_MAP_FILE.read_text(encoding="utf-8"))


def main():
    train, test = load_raw()
    train.to_csv(DATA_DIR / "train.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)
    print(f"Saved {len(train)} train and {len(test)} test rows to {DATA_DIR}")


if __name__ == "__main__":
    main()
