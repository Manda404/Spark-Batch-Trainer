"""Read-only training history exposed by all backends."""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class TrainingHistory:
    """Metric curves collected after every processed batch."""

    train_scores: Tuple[Tuple[float, ...], ...]
    validation_scores: Tuple[Tuple[float, ...], ...]
    batch_numbers: Tuple[int, ...]
    learning_rates: Tuple[float, ...] = ()

    @classmethod
    def from_lists(
        cls,
        train_scores: List[List[float]],
        validation_scores: List[List[float]],
        batch_numbers: List[int],
        learning_rates: List[float],
    ) -> "TrainingHistory":
        return cls(
            tuple(tuple(values) for values in train_scores),
            tuple(tuple(values) for values in validation_scores),
            tuple(batch_numbers),
            tuple(learning_rates),
        )
