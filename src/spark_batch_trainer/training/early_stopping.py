"""Backend-agnostic early stopping across training batches."""

from dataclasses import dataclass
from typing import Optional

from .metrics import MetricPolicy


@dataclass(frozen=True)
class EarlyStoppingDecision:
    improved: bool
    should_stop: bool
    patience_counter: int
    best_score: float
    improvement: Optional[float]


class GlobalEarlyStopping:
    """Compare one aggregate validation score after every batch."""

    @staticmethod
    def observe(
        *,
        current_score: float,
        best_score: Optional[float],
        metric_name: str,
        patience_counter: int,
        max_patience: int,
        mode: str = "auto",
        min_delta: float = 0.0,
    ) -> EarlyStoppingDecision:
        if max_patience < 1:
            raise ValueError("max_patience must be >= 1")

        improved = best_score is None or MetricPolicy.is_improvement(
            current_score,
            best_score,
            metric_name,
            mode,
            min_delta,
        )
        if improved:
            delta = None if best_score is None else abs(current_score - best_score)
            return EarlyStoppingDecision(True, False, 0, current_score, delta)

        next_patience = patience_counter + 1
        return EarlyStoppingDecision(
            False,
            next_patience >= max_patience,
            next_patience,
            best_score,
            None,
        )
