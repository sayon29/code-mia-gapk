import numpy as np

from code_mia.evaluation.metrics import calculate_metrics, empirical_tpr_at_fpr


def test_empirical_tpr_is_largest_point_without_interpolation():
    fpr = np.array([0.0, 0.01, 0.04, 0.2])
    tpr = np.array([0.1, 0.3, 0.7, 1.0])
    assert empirical_tpr_at_fpr(fpr, tpr, 0.05) == 0.7
    assert empirical_tpr_at_fpr(fpr, tpr, 0.001) == 0.1


def test_low_fpr_resolution_warning():
    labels = [0] * 100 + [1] * 100
    scores = list(np.linspace(-3, -2, 100)) + list(np.linspace(-1, 0, 100))
    metrics, _ = calculate_metrics(labels, scores)
    assert metrics["minimum_nonzero_empirical_fpr"] == 0.01
    assert any("not empirically resolvable" in warning for warning in metrics["warnings"])

