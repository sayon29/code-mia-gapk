from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_roc(curve: dict, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(curve["fpr"], curve["tpr"], label="Gap-K%")
    ax.plot([0, 1], [0, 1], "--", color="0.6", label="chance")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="ROC curve", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def plot_histogram(labels: list[int], scores: list[float], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    values = np.asarray(scores); y = np.asarray(labels)
    ax.hist(values[y == 0], bins="auto", alpha=0.65, label="non-member (0)")
    ax.hist(values[y == 1], bins="auto", alpha=0.65, label="member (1)")
    ax.set(xlabel="Gap-K% score (higher is more member-like)", ylabel="Count", title="Score distribution")
    ax.legend(); ax.grid(alpha=0.2); fig.tight_layout(); fig.savefig(output, dpi=160); plt.close(fig)


def write_summary_csv(rows: list[dict], output: Path) -> None:
    groups = {}
    for label in (0, 1):
        values = np.asarray([float(row["gapk_score"]) for row in rows if int(row["membership"]) == label])
        groups[label] = {"membership": label, "count": len(values), "mean": float(values.mean()),
                         "std": float(values.std()), "min": float(values.min()), "median": float(np.median(values)),
                         "max": float(values.max())}
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(groups[0]))
        writer.writeheader(); writer.writerows(groups.values())

