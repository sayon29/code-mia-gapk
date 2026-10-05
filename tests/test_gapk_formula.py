import math

import numpy as np
import torch

from code_mia.attacks.gapk import bottom_fraction_mean, moving_average_valid, token_gap_scores


def _independent_reference(row, target, floor=1e-8):
    values = np.asarray(row, dtype=np.float64)
    exp = np.exp(values - values.max()); p = exp / exp.sum(); logp = np.log(p)
    mu = np.sum(p * logp)
    var = np.sum(p * (logp - mu) ** 2)
    return (logp[target] - logp.max()) / math.sqrt(max(var, floor))


def test_hand_computed_full_vocabulary_formula_and_float32():
    logits = torch.tensor([[0.0, 1.0, 2.0], [2.0, -1.0, 0.5]], dtype=torch.float16)
    targets = torch.tensor([0, 2])
    actual = token_gap_scores(logits, targets)
    expected = [_independent_reference(row, target) for row, target in zip(logits.tolist(), targets.tolist())]
    assert actual.dtype == torch.float32
    assert np.allclose(actual.numpy(), expected, rtol=2e-5, atol=2e-5)


def test_variance_floor_prevents_nonfinite_values():
    actual = token_gap_scores(torch.zeros(2, 4), torch.tensor([0, 2]), variance_floor=1e-8)
    assert torch.isfinite(actual).all()
    assert actual.tolist() == [0.0, 0.0]


def test_valid_moving_average_values_and_length():
    result = moving_average_valid(torch.tensor([1.0, 2.0, 6.0, 3.0]), 3)
    assert np.allclose(result.numpy(), [3.0, 11.0 / 3.0])
    assert len(result) == 4 - 3 + 1


def test_bottom_fraction_floor_and_minimum_one():
    score, count = bottom_fraction_mean(torch.tensor([-1.0, -5.0, -2.0, -4.0, -3.0]), 0.20)
    assert count == 1 and score.item() == -5.0
    score, count = bottom_fraction_mean(torch.arange(11.0), 0.20)
    assert count == 2 and score.item() == 0.5


def test_official_commit_parity_translation_on_synthetic_logits():
    # Independent NumPy transcription of official commit 46cd478f's standardized
    # full-distribution gap; avoids downloading a model during tests.
    logits = torch.tensor([[1.2, -0.7, 0.3, 2.1]])
    assert np.isclose(token_gap_scores(logits, torch.tensor([2])).item(), _independent_reference(logits[0], 2), rtol=1e-5)
