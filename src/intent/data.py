"""Load and cache the Banking77 dataset.

Fetched as plain CSV from PolyAI's official task-specific-datasets repo. We deliberately avoid
HuggingFace `load_dataset` here: the `PolyAI/banking77` hub repo ships a loading *script* only,
and `datasets>=3` no longer executes scripts. The CSV is stable, tiny, and needs no extra deps.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
LABEL_MAP_FILE = DATA_DIR / "label_map.json"

TRAIN_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/train.csv"
)
TEST_URL = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/test.csv"
)


def _read_split(csv_path: Path, url: str) -> pd.DataFrame:
    if csv_path.exists():
        frame = pd.read_csv(csv_path)
    else:
        frame = pd.read_csv(url)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_csv(csv_path, index=False)
    # PolyAI's CSVs label the target column "category"; some mirrors use "intent".
    label_col = "intent" if "intent" in frame.columns else "category"
    return pd.DataFrame(
        {
            "text": frame["text"].astype(str).str.strip(),
            "label": frame[label_col].astype(str).str.strip(),
        }
    )


def load_raw(as_dataframe: bool = True):
    """Return (train_df, test_df) with columns text/label, downloading and caching on first use."""
    train = _read_split(DATA_DIR / "train.csv", TRAIN_URL)
    test = _read_split(DATA_DIR / "test.csv", TEST_URL)
    label_names = sorted(set(train["label"]) | set(test["label"]))
    _save_label_map(label_names)
    if not as_dataframe:
        raise ValueError("this loader is dataframe-only; call load_raw()")
    return train, test


def _save_label_map(label_names: list[str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_MAP_FILE.write_text(json.dumps(label_names, indent=2), encoding="utf-8")


def load_label_map() -> list[str]:
    return json.loads(LABEL_MAP_FILE.read_text(encoding="utf-8"))


def main():
    train, test = load_raw()
    print(f"Loaded {len(train)} train and {len(test)} test rows "
          f"({len(set(train['label']) | set(test['label']))} intents) -> {DATA_DIR}")


if __name__ == "__main__":
    main()
