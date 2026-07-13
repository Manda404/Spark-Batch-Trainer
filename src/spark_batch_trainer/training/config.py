"""Validated configuration shared by every training backend."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable options shared by every model backend."""

    num_batches: int = 10
    max_patience: int = 5
    use_sample_weight: bool = False
    verbose: bool = True
    show_learning_curve: bool = False
    metric_mode: str = "auto"
    min_delta: float = 0.0

    @classmethod
    def from_mapping(
        cls, values: Optional[Mapping[str, Any]]
    ) -> "TrainingConfig":
        """Build a validated config while ignoring backend-specific keys."""
        source = values or {}
        config = cls(
            num_batches=int(source.get("num_batches", 10)),
            max_patience=int(source.get("max_patience", 5)),
            use_sample_weight=bool(source.get("use_sample_weight", False)),
            verbose=bool(source.get("verbose", True)),
            show_learning_curve=bool(source.get("show_learning_curve", False)),
            metric_mode=str(source.get("metric_mode", "auto")).lower(),
            min_delta=float(source.get("min_delta", 0.0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """Reject invalid values before starting any Spark action."""
        if self.num_batches < 1:
            raise ValueError("num_batches must be >= 1")
        if self.max_patience < 1:
            raise ValueError("max_patience must be >= 1")
        if self.metric_mode not in {"auto", "min", "max"}:
            raise ValueError("metric_mode must be one of: 'auto', 'min', 'max'")
        if self.min_delta < 0:
            raise ValueError("min_delta must be >= 0")

    def apply_to(self, runtime: dict[str, Any]) -> None:
        """Populate the legacy runtime mapping during the compatibility phase."""
        runtime.update(
            {
                "num_batches": self.num_batches,
                "max_patience": self.max_patience,
                "use_sample_weight": self.use_sample_weight,
                "verbose": self.verbose,
                "show_learning_curve": self.show_learning_curve,
                "metric_mode": self.metric_mode,
                "min_delta": self.min_delta,
            }
        )
