"""Integration coverage shared by all supported model backends."""

from typing import Any

import pytest

from spark_batch_trainer import CatBoostTrainer, LightGBMTrainer, XGBoostTrainer
from spark_batch_trainer.training.base import BatchTrainer
from tests.data._data_mocks import assert_predicts_sample
from tests.processing.label_encoder import encode_target

BINARY_BACKENDS = [
    pytest.param(
        XGBoostTrainer,
        {"objective": "binary:logistic", "n_estimators": 10, "max_depth": 3},
        id="xgboost",
    ),
    pytest.param(
        LightGBMTrainer,
        {"objective": "binary", "n_estimators": 10, "max_depth": 3},
        id="lightgbm",
    ),
    pytest.param(
        CatBoostTrainer,
        {
            "loss_function": "Logloss",
            "iterations": 10,
            "depth": 3,
            "verbose": False,
        },
        id="catboost",
    ),
]

MULTICLASS_BACKENDS = [
    pytest.param(
        XGBoostTrainer,
        {"objective": "multi:softmax", "n_estimators": 10, "max_depth": 3},
        True,
        id="xgboost",
    ),
    pytest.param(
        LightGBMTrainer,
        {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "n_estimators": 10,
            "max_depth": 3,
        },
        True,
        id="lightgbm",
    ),
    pytest.param(
        CatBoostTrainer,
        {
            "loss_function": "MultiClass",
            "iterations": 10,
            "depth": 3,
            "verbose": False,
        },
        False,
        id="catboost",
    ),
]


@pytest.mark.parametrize(
    ("trainer_type", "backend_config"),
    BINARY_BACKENDS,
)
def test_backend_trains_binary_classifier(
    trainer_type: type[BatchTrainer[Any]],
    backend_config: dict[str, Any],
    mock_data_diabetes,
    config_training,
) -> None:
    train, valid, target = mock_data_diabetes
    trainer = trainer_type()

    trainer.fit(
        train,
        valid,
        target,
        training_config=config_training,
        model_config=dict(backend_config),
    )

    model = trainer.get_trained_model()
    assert model is not None
    assert trainer.get_training_history().batch_numbers == (1, 2)
    assert_predicts_sample(model, valid, target)
    inference_sample = valid.limit(5).toPandas()
    prepared_features = trainer.prepare_features(inference_sample)
    assert list(prepared_features.columns) == [
        name for name in valid.columns if name != target
    ]
    assert len(model.predict(prepared_features)) == len(inference_sample)


@pytest.mark.parametrize(
    ("trainer_type", "backend_config", "needs_num_class"),
    MULTICLASS_BACKENDS,
)
def test_backend_trains_multiclass_classifier(
    trainer_type: type[BatchTrainer[Any]],
    backend_config: dict[str, Any],
    needs_num_class: bool,
    mock_data_obesity,
    config_training,
) -> None:
    train, valid, target = mock_data_obesity
    encoded_train, encoded_valid, number_of_classes = encode_target(
        train, valid, target
    )
    model_config = dict(backend_config)
    if needs_num_class:
        model_config["num_class"] = number_of_classes
    trainer = trainer_type()

    trainer.fit(
        encoded_train,
        encoded_valid,
        target,
        training_config=config_training,
        model_config=model_config,
    )

    model = trainer.get_trained_model()
    assert model is not None
    assert trainer.get_training_history().batch_numbers == (1, 2)
    assert_predicts_sample(model, encoded_valid, target)
