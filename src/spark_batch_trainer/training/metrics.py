"""Metric direction policy independent from model frameworks."""


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
    def direction(cls, metric_name: str, mode: str = "auto") -> str:
        normalized_mode = mode.lower()
        if normalized_mode not in {"auto", "min", "max"}:
            raise ValueError("metric_mode must be one of: 'auto', 'min', 'max'")
        if normalized_mode != "auto":
            return normalized_mode

        metric = metric_name.lower().replace("-", "_").split(":", 1)[0]
        if metric.startswith(("auc", "map@", "ndcg@")):
            return "max"
        return "max" if metric in cls._MAXIMIZE_METRICS else "min"

    @classmethod
    def is_improvement(
        cls,
        current_score: float,
        best_score: float,
        metric_name: str,
        mode: str = "auto",
        min_delta: float = 0.0,
    ) -> bool:
        if min_delta < 0:
            raise ValueError("min_delta must be >= 0")
        if cls.direction(metric_name, mode) == "max":
            return current_score > best_score + min_delta
        return current_score < best_score - min_delta
