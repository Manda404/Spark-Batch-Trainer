"""Read-only training history exposed by all backends."""

from dataclasses import dataclass


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

    train_scores: tuple[tuple[float, ...], ...]
    validation_scores: tuple[tuple[float, ...], ...]
    batch_numbers: tuple[int, ...]
    learning_rates: tuple[float, ...] = ()
