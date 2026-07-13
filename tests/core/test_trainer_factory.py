import pytest

from spark_batch_trainer import create_trainer
from spark_batch_trainer.backends.catboost import CatBoostTrainer
from spark_batch_trainer.backends.lightgbm import LightGBMTrainer
from spark_batch_trainer.backends.xgboost import XGBoostTrainer


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("xgb", XGBoostTrainer),
        ("xgboost", XGBoostTrainer),
        ("catboost", CatBoostTrainer),
        ("lgbm", LightGBMTrainer),
        ("lightgbm", LightGBMTrainer),
    ],
)
def test_create_trainer(name: str, expected_type: type) -> None:
    assert isinstance(create_trainer(name), expected_type)


def test_create_trainer_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        create_trainer("random_forest")
