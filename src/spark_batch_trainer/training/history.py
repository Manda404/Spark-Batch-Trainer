"""Read-only training history exposed by all backends."""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class TrainingHistory:
    """Metric curves collected after every processed batch.

    Attributes:
        train_scores: One tuple of per-iteration training metric values per
            processed batch.
        validation_scores: One tuple of per-iteration validation metric
            values per processed batch.
        batch_numbers: One-based index of every processed batch, in order.
        learning_rates: Learning rate applied to each batch, or empty if no
            scheduler was configured.
    """

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
        """Build an immutable history from the mutable lists a trainer accumulates.

        Args:
            train_scores: Per-batch training metric values, in processing order.
            validation_scores: Per-batch validation metric values, in processing order.
            batch_numbers: One-based index of every processed batch.
            learning_rates: Learning rate applied to each batch, or an empty
                list when no scheduler was used.

        Returns:
            TrainingHistory: Immutable snapshot with every list converted to a tuple.
        """
        return cls(
            tuple(tuple(values) for values in train_scores),
            tuple(tuple(values) for values in validation_scores),
            tuple(batch_numbers),
            tuple(learning_rates),
        )
