"""Validated configuration shared by every training backend."""

from dataclasses import dataclass, fields
from typing import Any, Literal, Mapping, Optional, cast

MetricMode = Literal["auto", "min", "max"]


def _reject_unknown_keys(values: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        raise ValueError(f"unknown configuration option(s): {names}")


def _read_int(values: Mapping[str, Any], name: str, default: int) -> int:
    value = values.get(name, default)
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    return value


def _read_bool(values: Mapping[str, Any], name: str, default: bool) -> bool:
    value = values.get(name, default)
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")
    return value


def _read_float(values: Mapping[str, Any], name: str, default: float) -> float:
    value = values.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    return float(value)


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable options shared by every model backend.

    Attributes:
        num_batches (int): Number of target-stratified Spark batches to
            train on. Defaults to 10.
        max_patience (int): Consecutive non-improving batches allowed
            before early stopping. Defaults to 5.
        use_sample_weight (bool): Whether to apply class-balanced sample
            weights during training. Defaults to ``False``.
        show_learning_curve (bool): Whether to render the learning curve
            after training completes. Defaults to ``False``.
        metric_mode (str): Direction in which the monitored metric must
            improve — one of ``"auto"``, ``"min"``, ``"max"``. ``"auto"``
            infers the direction from the metric name. Defaults to ``"auto"``.
        min_delta (float): Minimum change required for a batch to count as
            an improvement. Defaults to ``0.0``.
        monitor_metric (str | None): Metric to record when a backend reports
            more than one. Defaults to ``None``.
    """

    num_batches: int = 10
    max_patience: int = 5
    use_sample_weight: bool = False
    show_learning_curve: bool = False
    metric_mode: MetricMode = "auto"
    min_delta: float = 0.0
    monitor_metric: Optional[str] = None

    @classmethod
    def from_mapping(
        cls, values: Optional[Mapping[str, Any] | "TrainingConfig"]
    ) -> "TrainingConfig":
        """Build a validated config from a raw mapping.

        ``values=None`` yields the default configuration. Unknown keys and
        ambiguous value types are rejected so configuration mistakes fail fast.

        Args:
            values: Raw options mapping, or ``None`` for the defaults.

        Returns:
            TrainingConfig: A validated configuration built from ``values``.

        Raises:
            ValueError: If a recognized value is out of its accepted range.
        """
        if isinstance(values, TrainingConfig):
            return values
        source = values or {}
        _reject_unknown_keys(source, {field.name for field in fields(cls)})
        metric_mode = source.get("metric_mode", "auto")
        if not isinstance(metric_mode, str):
            raise TypeError("metric_mode must be a string")
        monitor_metric = source.get("monitor_metric")
        if monitor_metric is not None and not isinstance(monitor_metric, str):
            raise TypeError("monitor_metric must be a string or None")
        return cls(
            num_batches=_read_int(source, "num_batches", 10),
            max_patience=_read_int(source, "max_patience", 5),
            use_sample_weight=_read_bool(source, "use_sample_weight", False),
            show_learning_curve=_read_bool(source, "show_learning_curve", False),
            metric_mode=cast(MetricMode, metric_mode.lower()),
            min_delta=_read_float(source, "min_delta", 0.0),
            monitor_metric=monitor_metric,
        )

    def __post_init__(self) -> None:
        """Reject invalid values as soon as the configuration is created.

        Raises:
            ValueError: If ``num_batches`` or ``max_patience`` is less than
                1, if ``metric_mode`` is not one of ``"auto"``, ``"min"``,
                ``"max"``, if ``min_delta`` is negative, or if
                ``monitor_metric`` is empty.
        """
        if self.num_batches < 1:
            raise ValueError("num_batches must be >= 1")
        if self.max_patience < 1:
            raise ValueError("max_patience must be >= 1")
        if self.metric_mode not in {"auto", "min", "max"}:
            raise ValueError("metric_mode must be one of: 'auto', 'min', 'max'")
        if self.min_delta < 0:
            raise ValueError("min_delta must be >= 0")
        if self.monitor_metric is not None and not self.monitor_metric.strip():
            raise ValueError("monitor_metric must not be empty")


@dataclass(frozen=True)
class LearningRateConfig:
    """Validated exponential-decay options used between training batches."""

    initial_lr: float = 0.1
    decay_rate: float = 0.95
    min_lr: float = 1e-4

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any] | "LearningRateConfig"
    ) -> "LearningRateConfig":
        """Build a strictly validated learning-rate configuration."""
        if isinstance(values, LearningRateConfig):
            return values
        _reject_unknown_keys(values, {field.name for field in fields(cls)})
        return cls(
            initial_lr=_read_float(values, "initial_lr", 0.1),
            decay_rate=_read_float(values, "decay_rate", 0.95),
            min_lr=_read_float(values, "min_lr", 1e-4),
        )

    def __post_init__(self) -> None:
        if self.initial_lr <= 0:
            raise ValueError("initial_lr must be > 0")
        if not 0 < self.decay_rate <= 1:
            raise ValueError("decay_rate must be in the interval (0, 1]")
        if self.min_lr <= 0:
            raise ValueError("min_lr must be > 0")
