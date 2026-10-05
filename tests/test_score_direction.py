import numpy as np
from sklearn.metrics import roc_auc_score

from code_mia.evaluation.metrics import calculate_metrics


def test_higher_scores_are_member_like_without_inversion():
    labels = [0, 0, 1, 1]
    scores = [-4.0, -3.0, -1.0, -0.2]
    metrics, _ = calculate_metrics(labels, scores)
    assert metrics["roc_auc"] == 1.0
    assert roc_auc_score(labels, scores) == 1.0
    assert "higher" in metrics["score_direction"]

