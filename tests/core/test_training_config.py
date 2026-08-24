import pytest

from spark_batch_trainer.training import LearningRateConfig, TrainingConfig
from spark_batch_trainer.training.state import TrainingRunState


def test_training_config_parses_legacy_mapping() -> None:
    config = TrainingConfig.from_mapping(
        {"num_batches": 4, "max_patience": 2, "metric_mode": "MAX"}
    )

    assert config.num_batches == 4
    assert config.max_patience == 2
    assert config.metric_mode == "max"


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"num_batches": 0}, "num_batches"),
        ({"max_patience": 0}, "max_patience"),
        ({"metric_mode": "largest"}, "metric_mode"),
        ({"min_delta": -1}, "min_delta"),
    ],
)
def test_training_config_rejects_invalid_values(values: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TrainingConfig.from_mapping(values)


def test_run_state_keeps_validated_config_and_backend_metric() -> None:
    state = TrainingRunState.from_mapping(
        {"num_batches": 3, "eval_metric": "auc"},
        default_eval_metric="logloss",
    )

    assert state.config.num_batches == 3
    assert state.eval_metric == "auc"
    assert state.best_model is None
    assert state.best_valid_score is None


@pytest.mark.parametrize(
    "values",
    [
        {"show_learning_curve": "false"},
        {"num_batches": 2.5},
        {"min_delta": "0.1"},
    ],
)
def test_training_config_rejects_ambiguous_types(values: dict) -> None:
    with pytest.raises(TypeError):
        TrainingConfig.from_mapping(values)


def test_training_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="max_pacience"):
        TrainingConfig.from_mapping({"max_pacience": 2})


def test_typed_configs_are_accepted_directly() -> None:
    training = TrainingConfig(num_batches=3)
    learning_rate = LearningRateConfig(initial_lr=0.05)

    assert TrainingConfig.from_mapping(training) is training
    assert LearningRateConfig.from_mapping(learning_rate) is learning_rate
