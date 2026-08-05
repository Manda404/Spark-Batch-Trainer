"""LightGBM implementation of the shared batch-training workflow."""

from typing import Any, Mapping, Optional

from lightgbm import LGBMClassifier
from pyspark.sql import DataFrame as SparkDataFrame

from spark_batch_trainer.data.spark_batching import iter_pandas_batches
from spark_batch_trainer.training.base import BatchTrainer
from spark_batch_trainer.training.state import PreparedDataset, TrainingRunState


class LightGBMTrainer(BatchTrainer[LGBMClassifier]):
    """Train a LightGBM classifier incrementally on stratified Spark batches."""

    def fit(
        self,
        train_dataframe: Optional[SparkDataFrame],
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
        **kwargs: Any,
    ) -> None:
        """Train the model and retain the best validation checkpoint."""
        model_config = dict(kwargs.get("model_config") or {})
        training_values: Optional[Mapping[str, Any]] = kwargs.get("training_config")
        learning_rate_config: Optional[Mapping[str, Any]] = kwargs.get(
            "learning_rate_config"
        )
        state = TrainingRunState[LGBMClassifier].from_mapping(
            training_values,
            default_eval_metric=str(model_config.get("metric", "binary_logloss")),
        )

        self._reset_run_history()
        train_dataframe, valid_dataframe = self._validate_inputs(
            train_dataframe,
            valid_dataframe,
            target_column,
            state.config.num_batches,
        )
        validation_data = self._prepare_validation(
            train_dataframe,
            valid_dataframe,
            target_column,
            use_sample_weight=state.config.use_sample_weight,
        )
        batches = iter_pandas_batches(
            train_dataframe,
            target_column,
            state.config.num_batches,
            self._logger,
        )
        self._logger.info(
            "Starting LightGBM training with %d batches", state.config.num_batches
        )

        batch_number = 0
        try:
            for batch_number, pandas_batch in enumerate(batches, start=1):
                self._logger.info(
                    "Processing batch %d/%d",
                    batch_number,
                    state.config.num_batches,
                )
                batch_data = self._prepare_batch(
                    pandas_batch,
                    target_column,
                    use_sample_weight=state.config.use_sample_weight,
                )
                learning_rate = self._resolve_learning_rate(
                    learning_rate_config,
                    batch_number,
                    default_lr=float(model_config.get("learning_rate", 0.1)),
                )
                model = self._fit_batch(
                    state,
                    batch_data,
                    validation_data,
                    model_config,
                    learning_rate,
                )
                if self._evaluate_model(
                    model, state, batch_number, framework="lightgbm"
                ):
                    self._logger.info("Global early stopping triggered")
                    break
        except Exception:
            self._logger.exception("LightGBM training failed at batch %d", batch_number)
            raise

        self._model = self._finalize_run(state, model_name="LightGBM")

    def _fit_batch(
        self,
        state: TrainingRunState[LGBMClassifier],
        batch_data: PreparedDataset,
        validation_data: PreparedDataset,
        model_config: Mapping[str, Any],
        learning_rate: float,
    ) -> LGBMClassifier:
        """Fit one batch, warm-started from the preceding LightGBM model."""
        model = LGBMClassifier(**{**model_config, "learning_rate": learning_rate})
        training_weight = (
            batch_data.sample_weight if state.config.use_sample_weight else None
        )
        validation_weight = (
            validation_data.sample_weight if state.config.use_sample_weight else None
        )
        model.fit(
            batch_data.features,
            batch_data.target,
            eval_set=[
                (batch_data.features, batch_data.target),
                (validation_data.features, validation_data.target),
            ],
            eval_names=["train", "valid"],
            eval_metric=state.eval_metric,
            categorical_feature=list(self._category_schema) or "auto",
            sample_weight=training_weight,
            eval_sample_weight=(
                [training_weight, validation_weight]
                if state.config.use_sample_weight
                else None
            ),
            init_model=state.previous_model,
        )
        return model
