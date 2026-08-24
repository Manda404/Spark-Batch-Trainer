"""Backend-agnostic early stopping across training batches."""

from dataclasses import dataclass
from logging import Logger
from math import isfinite
from typing import Optional

from .metrics import is_improvement


@dataclass(frozen=True)
class EarlyStoppingDecision:
    """Result of comparing one validation score with the running best."""

    improved: bool
    should_stop: bool
    patience_counter: int
    best_score: float
    improvement: Optional[float]


def observe_early_stopping(
    *,
    current_score: float,
    best_score: Optional[float],
    metric_name: str,
    patience_counter: int,
    max_patience: int,
    mode: str = "auto",
    min_delta: float = 0.0,
    logger: Optional[Logger] = None,
) -> EarlyStoppingDecision:
    """Return the early-stopping state after observing one batch score."""
    if max_patience < 1:
        raise ValueError("max_patience must be >= 1")
    if not isfinite(current_score):
        raise ValueError("current_score must be finite")
    if best_score is not None and not isfinite(best_score):
        raise ValueError("best_score must be finite")

    improved = best_score is None or is_improvement(
        current_score,
        best_score,
        metric_name,
        mode,
        min_delta,
        logger,
    )
    if improved:
        improvement = None if best_score is None else abs(current_score - best_score)
        return EarlyStoppingDecision(True, False, 0, current_score, improvement)

    assert best_score is not None
    next_patience = patience_counter + 1
    return EarlyStoppingDecision(
        False,
        next_patience >= max_patience,
        next_patience,
        best_score,
        None,
    )
