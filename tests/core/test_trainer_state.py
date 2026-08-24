import pytest

from spark_batch_trainer.training.base import BatchTrainer
from spark_batch_trainer.training.config import TrainingConfig
from spark_batch_trainer.training.state import TrainingRunState


class ConcreteTrainer(BatchTrainer):
    def __init__(self) -> None:
        super().__init__()
        self._global_train_loss = [[1.0]]
        self._global_valid_loss = [[2.0]]
        self._global_iterations = [1]
        self._lr_schedulers = [0.1]
        self._model = object()
        self._feature_columns = ("feature",)

    def fit(
        self, train_dataframe, valid_dataframe, target_column: str, **kwargs
    ) -> None:
        return None


def test_reset_run_history_clears_previous_fit_diagnostics() -> None:
    trainer = ConcreteTrainer()

    trainer._reset_run_history()

    assert trainer._global_train_loss == []
    assert trainer._global_valid_loss == []
    assert trainer._global_iterations == []
    assert trainer._lr_schedulers == []
    assert trainer.get_trained_model() is None
    assert trainer._feature_columns == ()


class MultiMetricXGBoostModel:
    def evals_result(self):
        return {
            "validation_0": {"logloss": [0.4], "auc": [0.8]},
            "validation_1": {"logloss": [0.5], "auc": [0.7]},
        }


def test_multiple_metrics_require_an_explicit_monitor() -> None:
    trainer = ConcreteTrainer()

    with pytest.raises(ValueError, match="monitor_metric"):
        trainer._extract_metric_history(MultiMetricXGBoostModel(), "xgboost")

    train, valid, metric = trainer._extract_metric_history(
        MultiMetricXGBoostModel(), "xgboost", monitor_metric="AUC"
    )
    assert train == [0.8]
    assert valid == [0.7]
    assert metric == "auc"


class XGBoostModel:
    def __init__(self, validation_score: float) -> None:
        self.validation_score = validation_score

    def evals_result(self):
        return {
            "validation_0": {"logloss": [self.validation_score - 0.1]},
            "validation_1": {"logloss": [self.validation_score]},
        }


def test_batch_level_early_stopping_selects_the_best_model() -> None:
    trainer = ConcreteTrainer()
    trainer._reset_run_history()
    state = TrainingRunState[object](
        config=TrainingConfig(max_patience=2, metric_mode="min"),
        eval_metric="logloss",
    )
    models = [XGBoostModel(score) for score in (0.5, 0.4, 0.41, 0.42)]

    decisions = [
        trainer._evaluate_model(model, state, batch_number, "xgboost")
        for batch_number, model in enumerate(models, start=1)
    ]

    assert decisions == [False, False, False, True]
    assert state.best_model is models[1]
    assert trainer._finalize_run(state, model_name="XGBoost") is models[1]
