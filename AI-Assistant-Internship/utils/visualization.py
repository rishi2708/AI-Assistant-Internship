"""Visualization helpers used by the dashboard."""

from __future__ import annotations

import io
from collections import Counter
from typing import Any


def sentiment_distribution(labels: list[str]):
    """Build a sentiment distribution bar chart."""

    import plotly.graph_objects as go

    counter = Counter(labels or ["Neutral"])
    names = list(counter.keys())
    values = [counter[name] for name in names]
    colors = {"Positive": "#2ca02c", "Neutral": "#7f7f7f", "Negative": "#d62728"}
    return go.Figure(
        data=[go.Bar(x=names, y=values, marker_color=[colors.get(name, "#00d4ff") for name in names])],
        layout=go.Layout(title="Sentiment Distribution", xaxis_title="Sentiment", yaxis_title="Count"),
    )


def latency_chart(latencies: list[float]):
    """Build response latency line chart."""

    import plotly.graph_objects as go

    values = latencies or [0.0]
    return go.Figure(
        data=[go.Scatter(x=list(range(1, len(values) + 1)), y=values, mode="lines+markers")],
        layout=go.Layout(title="Response Latency", xaxis_title="Turn", yaxis_title="Latency (s)"),
    )


def dataset_statistics_chart(stats: dict[str, int]):
    """Build dataset statistics chart."""

    import plotly.graph_objects as go

    return go.Figure(
        data=[go.Bar(x=list(stats.keys()), y=list(stats.values()), marker_color="#00d4ff")],
        layout=go.Layout(title="Dataset Statistics", xaxis_title="Dataset", yaxis_title="Documents"),
    )


def embedding_projection(embeddings, labels: list[str] | None = None) -> Any:
    """Project embeddings to two dimensions using PCA."""

    import numpy as np
    import pandas as pd
    import plotly.express as px
    from sklearn.decomposition import PCA

    if embeddings.size == 0:
        embeddings = np.zeros((1, 2))
    if embeddings.shape[0] == 1:
        points = np.array([[0.0, 0.0]])
    else:
        points = PCA(n_components=2, random_state=42).fit_transform(embeddings)
    labels = labels or [f"Doc {index + 1}" for index in range(points.shape[0])]
    frame = pd.DataFrame({"x": points[:, 0], "y": points[:, 1], "label": labels[: points.shape[0]]})
    return px.scatter(frame, x="x", y="y", hover_name="label", title="Embedding Visualization")


def confusion_matrix_png(matrix, labels: list[str]) -> bytes:
    """Render a confusion matrix as PNG bytes."""

    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, matrix[row, col], ha="center", va="center", color="black")
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return buffer.getvalue()
