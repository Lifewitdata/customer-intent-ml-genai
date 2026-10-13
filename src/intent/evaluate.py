"""Classification metrics."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score


def report(y_true, y_pred, label_names: list[str] | None = None) -> dict:
    """Return a concise metrics dict plus a per-class DataFrame."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class": pd.DataFrame(
            classification_report(
                y_true, y_pred, target_names=label_names, output_dict=True, zero_division=0
            )
        ).T,
    }


def print_report(metrics: dict) -> None:
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Macro F1 : {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
