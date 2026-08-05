from spark_batch_trainer import LightGBMTrainer, XGBoostTrainer


def test_xgboost_uses_model_learning_rate_without_scheduler() -> None:
    trainer = XGBoostTrainer()

    learning_rate = trainer._resolve_learning_rate(None, batch_id=1, default_lr=0.025)

    assert learning_rate == 0.025


def test_lightgbm_uses_model_learning_rate_without_scheduler() -> None:
    trainer = LightGBMTrainer()

    learning_rate = trainer._resolve_learning_rate(None, batch_id=1, default_lr=0.025)

    assert learning_rate == 0.025
