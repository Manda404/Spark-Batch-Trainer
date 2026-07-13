import pytest

from spark_batch_trainer.training import TrainingConfig


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


def test_apply_to_preserves_backend_specific_options() -> None:
    runtime = {"eval_metric": "auc"}

    TrainingConfig(num_batches=3).apply_to(runtime)

    assert runtime["eval_metric"] == "auc"
    assert runtime["num_batches"] == 3
