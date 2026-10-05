"""Exact vanilla Gap-K% statistics over the complete vocabulary."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GapKResult:
    score: float
    raw_count: int
    smoothed_count: int
    selected_count: int
    gaps: torch.Tensor
    smoothed: torch.Tensor


def causal_logits_and_targets(logits: torch.Tensor, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Align logits at t-1 with true tokens at t, excluding the first token."""
    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("Expected logits [batch, sequence, vocab] and input_ids [batch, sequence]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("Logit and token batch/sequence dimensions differ")
    return logits[:, :-1, :], input_ids[:, 1:]


def token_gap_scores(logits: torch.Tensor, targets: torch.Tensor, variance_floor: float = 1e-8) -> torch.Tensor:
    """Compute standardized true-vs-top1 gaps using float32 full-vocabulary statistics."""
    if logits.ndim < 2 or logits.shape[:-1] != targets.shape:
        raise ValueError("targets must match every logits dimension except vocabulary")
    logp = torch.log_softmax(logits.float(), dim=-1)
    probabilities = logp.exp()
    true_logp = logp.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1)
    max_logp = logp.max(dim=-1).values
    # Reuse the probability buffer for the first and second raw moments. Since
    # Var(log p) = E[(log p)^2] - E[log p]^2, this is algebraically identical
    # to the centered definition while retaining only two full-vocabulary
    # float32 tensors (logp and the reused work buffer).
    probabilities.mul_(logp)
    mean = probabilities.sum(dim=-1)
    probabilities.mul_(logp)
    second_moment = probabilities.sum(dim=-1)
    variance = second_moment - mean.square()
    sigma = variance.clamp_min(float(variance_floor)).sqrt()
    gaps = (true_logp - max_logp) / sigma
    if not torch.isfinite(gaps).all():
        raise FloatingPointError("Non-finite Gap-K token statistic detected")
    return gaps


def moving_average_valid(values: torch.Tensor, window_size: int) -> torch.Tensor:
    if values.ndim != 1:
        raise ValueError("moving average expects a one-dimensional tensor")
    if window_size < 1:
        raise ValueError("window_size must be positive")
    if values.numel() < window_size:
        raise ValueError(f"Need at least {window_size} token gaps; got {values.numel()}")
    return values.unfold(0, window_size, 1).mean(dim=-1)


def bottom_fraction_mean(values: torch.Tensor, fraction: float) -> tuple[torch.Tensor, int]:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("bottom-fraction selection requires a non-empty vector")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    count = max(1, math.floor(float(fraction) * values.numel()))
    selected = torch.sort(values).values[:count]
    return selected.mean(), count


def score_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    gap_fraction: float = 0.20,
    smoothing_window: int = 3,
    variance_floor: float = 1e-8,
) -> GapKResult:
    shifted_logits, targets = causal_logits_and_targets(logits, input_ids)
    if shifted_logits.shape[0] != 1:
        raise ValueError("The baseline supports one sample at a time")
    gaps = token_gap_scores(shifted_logits[0], targets[0], variance_floor)
    smoothed = moving_average_valid(gaps, smoothing_window)
    score, count = bottom_fraction_mean(smoothed, gap_fraction)
    return GapKResult(float(score.item()), gaps.numel(), smoothed.numel(), count, gaps, smoothed)
