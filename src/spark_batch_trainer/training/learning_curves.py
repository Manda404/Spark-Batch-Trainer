"""Optional learning-curve rendering."""

from itertools import accumulate


def plot_metrics(
    train_scores: list[list[float]],
    validation_scores: list[list[float]],
    batch_numbers: list[int],
    model_name: str,
    metric_name: str,
) -> None:
    """Plot flattened train and validation curves with batch boundaries."""
    from matplotlib import pyplot

    flattened_train = [value for curve in train_scores for value in curve]
    flattened_validation = [value for curve in validation_scores for value in curve]
    boundaries = accumulate([0] + [len(curve) for curve in train_scores[:-1]])

    _, axis = pyplot.subplots(figsize=(20, 6))
    axis.plot(flattened_train, label=f"Train {metric_name}")
    axis.plot(flattened_validation, label=f"Validation {metric_name}")
    for boundary, batch_number in zip(boundaries, batch_numbers):
        axis.axvline(boundary, color="red", linestyle="--", linewidth=0.8)
        axis.text(
            boundary,
            0.95,
            f"Batch {batch_number}",
            transform=axis.get_xaxis_transform(),
            ha="center",
            color="gray",
        )
    axis.set_title(f"Learning curve — {model_name}")
    axis.set_xlabel("Cumulative boosting iteration")
    axis.set_ylabel(metric_name)
    axis.legend()
    axis.grid(True)
    pyplot.show()


def plot_learning_rates(name: str, values: list[float]) -> None:
    """Plot one learning-rate value per processed batch."""
    if not values:
        return

    from matplotlib import pyplot

    batch_numbers = list(range(1, len(values) + 1))
    _, axis = pyplot.subplots(figsize=(16, 4))
    axis.plot(batch_numbers, values, marker="o", label=name)
    axis.set_title(f"Learning-rate schedule — {name}")
    axis.set_xlabel("Batch")
    axis.set_ylabel("Learning rate")
    axis.legend()
    axis.grid(True)
    pyplot.show()
