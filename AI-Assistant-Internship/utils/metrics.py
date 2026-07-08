"""Evaluation and latency metrics."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class MetricsTracker:
    """Track runtime metrics for UI visualizations."""

    latencies: list[float] = field(default_factory=list)
    sentiment_labels: list[str] = field(default_factory=list)

    @contextmanager
    def track_latency(self) -> Iterator[None]:
        """Context manager that records elapsed seconds."""

        started = time.perf_counter()
        try:
            yield
        finally:
            self.latencies.append(time.perf_counter() - started)

    def add_sentiment(self, label: str) -> None:
        """Record a sentiment label."""

        self.sentiment_labels.append(label)

    def latency_summary(self) -> dict[str, float]:
        """Return latency summary statistics."""

        if not self.latencies:
            return {"count": 0, "mean": 0.0, "p95": 0.0}
        values = sorted(self.latencies)
        p95_index = max(int(0.95 * (len(values) - 1)), 0)
        return {
            "count": float(len(values)),
            "mean": float(sum(values) / len(values)),
            "p95": float(values[p95_index]),
        }


def classification_report_dict(y_true: list[str], y_pred: list[str]) -> dict[str, object]:
    """Compute accuracy, precision, recall, F1, and confusion matrix."""

    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    return {
        "labels": labels,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }
