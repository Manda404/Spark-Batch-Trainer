"""Metric direction policy independent from model frameworks."""

from logging import Logger
from typing import Optional


class MetricPolicy:
    """Determine how a named metric must be compared."""

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

    @classmethod
    def direction(
        cls, metric_name: str, mode: str = "auto", logger: Optional[Logger] = None
    ) -> str:
        """Resolve whether a metric must be minimized or maximized.

        ``mode="auto"`` infers the direction from ``metric_name`` (e.g.
        ``"auc"`` maximizes, ``"logloss"`` minimizes). Metrics outside the
        known list silently default to ``"min"``, which is wrong for a
        custom metric that should be maximized — pass ``logger`` to surface
        that fallback instead of failing silently.

        Args:
            metric_name: Metric name reported by the backend.
            mode: One of ``"auto"``, ``"min"``, ``"max"``. Defaults to ``"auto"``.
            logger: Optional logger used to warn when ``mode="auto"`` could
                not recognize ``metric_name`` and defaulted to ``"min"``.

        Returns:
            str: ``"min"`` or ``"max"``.

        Raises:
            ValueError: If ``mode`` is not one of ``"auto"``, ``"min"``, ``"max"``.
        """
        normalized_mode = mode.lower()
        if normalized_mode not in {"auto", "min", "max"}:
            raise ValueError("metric_mode must be one of: 'auto', 'min', 'max'")
        if normalized_mode != "auto":
            return normalized_mode

        metric = metric_name.lower().replace("-", "_").split(":", 1)[0]
        if metric.startswith(("auc", "map@", "ndcg@")):
            return "max"
        if metric in cls._MAXIMIZE_METRICS:
            return "max"
        if logger is not None:
            logger.warning(
                "metric_mode='auto' does not recognize metric %r; defaulting to "
                "'min'. Pass metric_mode='max' explicitly if this metric should "
                "be maximized, otherwise early stopping and best-model selection "
                "will run in the wrong direction.",
                metric_name,
            )
        return "min"

    @classmethod
    def is_improvement(
        cls,
        current_score: float,
        best_score: float,
        metric_name: str,
        mode: str = "auto",
        min_delta: float = 0.0,
        logger: Optional[Logger] = None,
    ) -> bool:
        """Decide whether ``current_score`` improves on ``best_score`` by ``min_delta``.

        Direction (min vs. max) is resolved via :meth:`direction`.

        Args:
            current_score: Score to compare against ``best_score``.
            best_score: Best score observed so far.
            metric_name: Metric name, passed through to :meth:`direction`.
            mode: One of ``"auto"``, ``"min"``, ``"max"``. Defaults to ``"auto"``.
            min_delta: Minimum change required to count as an improvement.
                Defaults to ``0.0``.
            logger: Optional logger passed through to :meth:`direction`.

        Returns:
            bool: ``True`` if ``current_score`` improves on ``best_score``
            by more than ``min_delta``.

        Raises:
            ValueError: If ``min_delta`` is negative.
        """
        if min_delta < 0:
            raise ValueError("min_delta must be >= 0")
        if cls.direction(metric_name, mode, logger) == "max":
            return current_score > best_score + min_delta
        return current_score < best_score - min_delta
