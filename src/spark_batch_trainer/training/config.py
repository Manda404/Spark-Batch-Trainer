"""Validated configuration shared by every training backend."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional


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
    """

    num_batches: int = 10
    max_patience: int = 5
    use_sample_weight: bool = False
    show_learning_curve: bool = False
    metric_mode: str = "auto"
    min_delta: float = 0.0

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]]) -> "TrainingConfig":
        """Build a validated config from a raw mapping, ignoring unknown keys.

        ``values=None`` yields the default configuration. Unknown keys are
        ignored so the same mapping can be passed across every backend.

        Args:
            values: Raw options mapping, or ``None`` for the defaults.

        Returns:
            TrainingConfig: A validated configuration built from ``values``.

        Raises:
            ValueError: If a recognized value is out of its accepted range.
        """
        source = values or {}
        return cls(
            num_batches=int(source.get("num_batches", 10)),
            max_patience=int(source.get("max_patience", 5)),
            use_sample_weight=bool(source.get("use_sample_weight", False)),
            show_learning_curve=bool(source.get("show_learning_curve", False)),
            metric_mode=str(source.get("metric_mode", "auto")).lower(),
            min_delta=float(source.get("min_delta", 0.0)),
        )

    def __post_init__(self) -> None:
        """Reject invalid values as soon as the configuration is created.

        Raises:
            ValueError: If ``num_batches`` or ``max_patience`` is less than
                1, if ``metric_mode`` is not one of ``"auto"``, ``"min"``,
                ``"max"``, or if ``min_delta`` is negative.
        """
        if self.num_batches < 1:
            raise ValueError("num_batches must be >= 1")
        if self.max_patience < 1:
            raise ValueError("max_patience must be >= 1")
        if self.metric_mode not in {"auto", "min", "max"}:
            raise ValueError("metric_mode must be one of: 'auto', 'min', 'max'")
        if self.min_delta < 0:
            raise ValueError("min_delta must be >= 0")
