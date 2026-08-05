"""Metric comparison rules used by global early stopping."""

from logging import Logger
from typing import Optional

_MAXIMIZE_METRICS = frozenset(
    {
        "acc",
        "accuracy",
        "auc",
        "aucpr",
        "average_precision",
        "map",
        "ndcg",
        "precision",
        "r2",
        "recall",
    }
)


def metric_direction(
    metric_name: str,
    mode: str = "auto",
    logger: Optional[Logger] = None,
) -> str:
    """Return ``"min"`` or ``"max"`` for a metric and explicit mode."""
    normalized_mode = mode.lower()
    if normalized_mode not in {"auto", "min", "max"}:
        raise ValueError("metric_mode must be one of: 'auto', 'min', 'max'")
    if normalized_mode != "auto":
        return normalized_mode

    metric = metric_name.lower().replace("-", "_").split(":", 1)[0]
    if metric.startswith(("auc", "map@", "ndcg@")) or metric in _MAXIMIZE_METRICS:
        return "max"
    if logger is not None:
        logger.warning(
            "Unknown metric %r; metric_mode='auto' defaults to 'min'", metric_name
        )
    return "min"


def is_improvement(
    current_score: float,
    best_score: float,
    metric_name: str,
    mode: str = "auto",
    min_delta: float = 0.0,
    logger: Optional[Logger] = None,
) -> bool:
    """Return whether ``current_score`` improves on ``best_score``."""
    if min_delta < 0:
        raise ValueError("min_delta must be >= 0")
    if metric_direction(metric_name, mode, logger) == "max":
        return current_score > best_score + min_delta
    return current_score < best_score - min_delta
