# ==========================================================
# tests/test_lightgbm_trainer_instantiation.py
# ==========================================================
# Purpose:
# This pytest test verifies the correct instantiation of the
# `LightGBMTrainer` class, which wraps a LightGBM model
# and provides a training interface.
#
# Scope of this test:
#   1. Ensure the object can be created without errors.
#   2. Ensure that minimal required attributes/methods exist:
#      - `fit` (main training entry point)
#      - `get_trained_model` (accessor for the underlying model)
#
# Note: We are not testing model training behavior here
# (fit logic, predictions, evaluation).
# Those will be covered in dedicated training tests.
# ==========================================================

import pytest
from spark_batch_trainer import LightGBMTrainer


@pytest.fixture
def trainer():
    """Fixture: create a LightGBMTrainer instance."""
    return LightGBMTrainer()


def test_instantiated(trainer):
    """
    Ensure the object is instantiated successfully.
    """
    assert trainer is not None


def test_has_minimal_attrs(trainer):
    """
    Ensure the object has the minimal required methods.
    """
    assert hasattr(trainer, "fit") and callable(trainer.fit)
    assert hasattr(trainer, "get_trained_model") and callable(trainer.get_trained_model)
