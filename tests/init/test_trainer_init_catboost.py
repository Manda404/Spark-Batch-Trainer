# ==========================================================
# tests/test_catboost_trainer_instantiation.py
# ==========================================================
# Purpose:
# This pytest test verifies the correct instantiation of the
# `CatBoostTrainer` class, which encapsulates a CatBoost model
# and provides a training interface.
#
# Scope of this test:
#   1. Ensure the object can be created without errors.
#   2. Ensure that minimal required attributes/methods exist:
#      - `fit` (main training entry point)
#      - `get_trained_model` (accessor for the underlying model)
#
# Note: We are not testing training behavior here
# (fit logic, predictions, evaluation, etc.).
# Those are handled in other dedicated tests.
# ==========================================================

import pytest
from spark_batch_trainer import CatBoostTrainer


@pytest.fixture
def trainer():
    """Fixture: create a CatBoostTrainer instance."""
    return CatBoostTrainer()


def test_instantiated(trainer):
    """
    Ensure the object is instantiated successfully.

    Expected:
    - `trainer` should not be None.
    """
    assert trainer is not None


def test_has_minimal_attrs(trainer):
    """
    Ensure the object has the minimal required methods.

    Expected:
    - `fit` must exist and be callable.
    - `get_trained_model` must exist and be callable.
    """
    assert hasattr(trainer, "fit") and callable(trainer.fit)
    assert hasattr(trainer, "get_trained_model") and callable(trainer.get_trained_model)
