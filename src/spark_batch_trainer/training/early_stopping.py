"""Backend-agnostic early stopping across training batches."""

from dataclasses import dataclass
from logging import Logger
from typing import Optional

from .metrics import MetricPolicy


@dataclass(frozen=True)
class EarlyStoppingDecision:
    """Outcome of comparing one batch's validation score to the running best.

    Attributes:
        improved: Whether ``current_score`` improved on the previous best.
        should_stop: Whether the patience budget is exhausted and training
            should stop.
        patience_counter: Updated count of consecutive non-improving batches.
        best_score: Best score known after this observation.
        improvement: Absolute improvement over the previous best, or
            ``None`` when this is the first score or there was no improvement.
    """

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
        logger: Optional[Logger] = None,
    ) -> EarlyStoppingDecision:
        """Update the early-stopping state with one batch's validation score.

        Args:
            current_score: Validation score for the batch just completed.
            best_score: Best score so far, or ``None`` before the first batch.
            metric_name: Used to resolve the comparison direction when
                ``mode="auto"``.
            patience_counter: Number of consecutive non-improving batches
                observed so far.
            max_patience: Non-improving batches allowed before stopping.
                Must be >= 1.
            mode: One of ``"auto"``, ``"min"``, ``"max"``. Defaults to ``"auto"``.
            min_delta: Minimum change required to count as an improvement.
                Defaults to ``0.0``.
            logger: Optional logger used to warn when ``mode="auto"`` could
                not recognize ``metric_name``; see
                :meth:`~spark_batch_trainer.training.metrics.MetricPolicy.direction`.

        Returns:
            EarlyStoppingDecision: The updated early-stopping state.

        Raises:
            ValueError: If ``max_patience`` is less than 1.
        """
        if max_patience < 1:
            raise ValueError("max_patience must be >= 1")

        improved = best_score is None or MetricPolicy.is_improvement(
            current_score,
            best_score,
            metric_name,
            mode,
            min_delta,
            logger,
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
