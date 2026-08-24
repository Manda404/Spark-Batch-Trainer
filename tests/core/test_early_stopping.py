import pytest

from spark_batch_trainer.training.early_stopping import observe_early_stopping


def test_first_observation_becomes_best() -> None:
    decision = observe_early_stopping(
        current_score=0.7,
        best_score=None,
        metric_name="auc",
        patience_counter=0,
        max_patience=2,
    )

    assert decision.improved
    assert decision.best_score == 0.7
    assert decision.patience_counter == 0


def test_patience_stops_after_consecutive_non_improvements() -> None:
    decision = observe_early_stopping(
        current_score=0.6,
        best_score=0.7,
        metric_name="auc",
        patience_counter=1,
        max_patience=2,
    )

    assert not decision.improved
    assert decision.should_stop
    assert decision.patience_counter == 2


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_score_is_rejected(score: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        observe_early_stopping(
            current_score=score,
            best_score=None,
            metric_name="logloss",
            patience_counter=0,
            max_patience=2,
        )
