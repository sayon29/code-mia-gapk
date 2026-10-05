from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def empirical_tpr_at_fpr(fpr: np.ndarray, tpr: np.ndarray, alpha: float) -> float:
    eligible = np.asarray(tpr)[np.asarray(fpr) <= alpha]
    return float(eligible.max()) if eligible.size else 0.0


def calculate_metrics(labels: Iterable[int], scores: Iterable[float], failures: int = 0) -> tuple[dict, dict]:
    y = np.asarray(list(labels), dtype=np.int64)
    s = np.asarray(list(scores), dtype=np.float64)
    if y.size == 0 or np.unique(y).size != 2:
        raise ValueError("Metrics require successful scores from both classes")
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
    non_members = int((y == 0).sum())
    min_fpr = 1.0 / non_members if non_members else None
    warnings: list[str] = []
    if min_fpr is not None and min_fpr > 0.001:
        warnings.append(
            f"TPR@0.1% FPR is not empirically resolvable with {non_members} non-members; "
            f"minimum nonzero empirical FPR is {min_fpr:.6g}. The reported value is the "
            "largest observed TPR at FPR <= 0.001 (usually the zero-FP operating point)."
        )
    metrics = {
        "roc_auc": float(roc_auc_score(y, s)),
        "tpr_at_fpr": {
            "0.05": empirical_tpr_at_fpr(fpr, tpr, 0.05),
            "0.01": empirical_tpr_at_fpr(fpr, tpr, 0.01),
            "0.001": empirical_tpr_at_fpr(fpr, tpr, 0.001),
        },
        "member_count": int((y == 1).sum()),
        "non_member_count": non_members,
        "successful_score_count": int(y.size),
        "failure_exclusion_count": int(failures),
        "minimum_nonzero_empirical_fpr": min_fpr,
        "score_direction": "higher (closer to zero) is more member-like; membership=1 is positive",
        "warnings": warnings,
    }
    curve = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thresholds.tolist()}
    return metrics, curve
