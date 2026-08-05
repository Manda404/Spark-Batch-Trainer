"""Class-balanced sample weights."""

from typing import Any

from numpy import asarray, bincount, float64, unique
from numpy.typing import NDArray
from pandas import isna


def calculate_sample_weights(target_values: Any) -> NDArray[float64]:
    """Return inverse-frequency weights normalized to a mean of one."""
    target = asarray(target_values).ravel()
    if target.size == 0:
        return asarray([], dtype=float64)
    if isna(target).any():
        raise ValueError("target values must not contain missing values")

    _, class_indices = unique(target, return_inverse=True)
    counts = bincount(class_indices)
    class_weights = target.size / (len(counts) * counts)
    weights = class_weights[class_indices].astype(float64, copy=False)
    weights /= weights.mean()
    return weights
