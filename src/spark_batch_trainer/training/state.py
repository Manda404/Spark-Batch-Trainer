"""Typed data exchanged by the shared training workflow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Mapping, Optional, TypeVar

from numpy import float64
from numpy.typing import NDArray
from pandas import DataFrame, Series

from .config import TrainingConfig

ModelT = TypeVar("ModelT")


@dataclass(frozen=True)
class PreparedDataset:
    """Pandas features, target, and weights ready for a model backend."""

    features: DataFrame
    target: Series[Any]
    sample_weight: Optional[NDArray[float64]] = None


@dataclass
class TrainingRunState(Generic[ModelT]):
    """Mutable state for one call to a trainer's ``fit`` method.

    Keeping these values in a dataclass makes the training loop explicit and
    prevents misspelled dictionary keys from failing only after a long Spark
    job has started.
    """

    config: TrainingConfig
    eval_metric: str
    previous_model: Optional[ModelT] = None
    best_model: Optional[ModelT] = None
    patience_counter: int = 0
    best_valid_score: Optional[float] = None
    observed_metric: Optional[str] = None

    @classmethod
    def from_mapping(
        cls,
        values: Optional[Mapping[str, Any] | TrainingConfig],
        *,
        default_eval_metric: str,
    ) -> "TrainingRunState[ModelT]":
        """Build validated run state while retaining a backend metric option."""
        if isinstance(values, TrainingConfig):
            return cls(config=values, eval_metric=default_eval_metric)
        source = dict(values or {})
        eval_metric = source.pop("eval_metric", default_eval_metric)
        if not isinstance(eval_metric, str):
            raise TypeError("eval_metric must be a string")
        return cls(
            config=TrainingConfig.from_mapping(source),
            eval_metric=eval_metric,
        )
