"""Tests for class-balanced sample weights."""

import numpy as np
import pytest

from spark_batch_trainer.data.sample_weighting import calculate_sample_weights


def test_balanced_weights_favor_the_minority_and_have_unit_mean() -> None:
    weights = calculate_sample_weights([0, 0, 0, 1])

    assert weights.mean() == pytest.approx(1.0)
    assert weights[-1] > weights[0]


def test_nan_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing"):
        calculate_sample_weights(np.array([0.0, np.nan]))
