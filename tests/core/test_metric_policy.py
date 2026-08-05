import pytest

from spark_batch_trainer.training.base import BatchTrainer
from spark_batch_trainer.training.metrics import is_improvement, metric_direction


@pytest.mark.parametrize("metric", ["auc", "AUC", "aucpr", "accuracy", "ndcg@10"])
def test_auto_mode_maximizes_score_metrics(metric: str) -> None:
    assert is_improvement(0.9, 0.8, metric)
    assert not is_improvement(0.7, 0.8, metric)


@pytest.mark.parametrize("metric", ["logloss", "multi_logloss", "rmse", "error"])
def test_auto_mode_minimizes_loss_metrics(metric: str) -> None:
    assert is_improvement(0.2, 0.3, metric)
    assert not is_improvement(0.4, 0.3, metric)


def test_explicit_mode_overrides_metric_name() -> None:
    assert is_improvement(0.9, 0.8, "custom", mode="max")
    assert is_improvement(0.2, 0.3, "custom", mode="min")


def test_min_delta_rejects_insignificant_change() -> None:
    assert not is_improvement(0.299, 0.3, "logloss", min_delta=0.01)


@pytest.mark.parametrize("mode", ["minimum", "maximum", "invalid"])
def test_invalid_metric_mode_is_rejected(mode: str) -> None:
    with pytest.raises(ValueError, match="metric_mode"):
        metric_direction("logloss", mode)


def test_negative_min_delta_is_rejected() -> None:
    with pytest.raises(ValueError, match="min_delta"):
        is_improvement(0.2, 0.3, "logloss", min_delta=-0.1)


class ConcreteTrainer(BatchTrainer):
    def __init__(self) -> None:
        super().__init__()
        self._global_train_loss = [[1.0]]
        self._global_valid_loss = [[2.0]]
        self._global_iterations = [1]
        self._lr_schedulers = [0.1]

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
