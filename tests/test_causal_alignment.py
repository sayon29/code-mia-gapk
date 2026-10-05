import torch

from code_mia.attacks.gapk import causal_logits_and_targets, score_from_logits


def test_alignment_excludes_first_token():
    ids = torch.tensor([[7, 4, 2, 9]])
    logits = torch.arange(1 * 4 * 10).reshape(1, 4, 10).float()
    shifted, targets = causal_logits_and_targets(logits, ids)
    assert torch.equal(shifted, logits[:, :3])
    assert targets.tolist() == [[4, 2, 9]]


def test_score_counts_only_predicted_tokens():
    ids = torch.tensor([[0, 1, 2, 3, 4]])
    logits = torch.randn(1, 5, 6)
    result = score_from_logits(logits, ids, smoothing_window=2)
    assert result.raw_count == 4
    assert result.smoothed_count == 3

