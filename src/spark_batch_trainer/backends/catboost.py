"""CatBoost implementation of the shared batch-training workflow."""

from typing import Any, Mapping, Optional

from catboost import CatBoostClassifier, Pool
from pyspark.sql import DataFrame as SparkDataFrame

from spark_batch_trainer.data.spark_batching import iter_pandas_batches
from spark_batch_trainer.training.base import BatchTrainer
from spark_batch_trainer.training.state import PreparedDataset, TrainingRunState


class CatBoostTrainer(BatchTrainer[CatBoostClassifier]):
    """Train a CatBoost classifier incrementally on stratified Spark batches."""

    def fit(
        self,
        train_dataframe: Optional[SparkDataFrame],
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
        **kwargs: Any,
    ) -> None:
        """Train the model and retain the best validation checkpoint."""
        model_config = dict(kwargs.get("model_config") or {})
        model_config.setdefault("allow_writing_files", False)
        training_values: Optional[Mapping[str, Any]] = kwargs.get("training_config")
        state = TrainingRunState[CatBoostClassifier].from_mapping(
            training_values,
            default_eval_metric=str(
                model_config.get(
                    "eval_metric", model_config.get("loss_function", "Logloss")
                )
            ),
        )

        self._reset_run_history()
        if kwargs.get("learning_rate_config") is not None:
            self._logger.warning(
                "CatBoost does not support the shared learning-rate scheduler; "
                "learning_rate_config is ignored"
            )
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
            "Starting CatBoost training with %d batches", state.config.num_batches
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
                model = self._fit_batch(
                    state, batch_data, validation_data, model_config
                )
                if self._evaluate_model(
                    model, state, batch_number, framework="catboost"
                ):
                    self._logger.info("Global early stopping triggered")
                    break
        except Exception:
            self._logger.exception("CatBoost training failed at batch %d", batch_number)
            raise

        self._model = self._finalize_run(state, model_name="CatBoost")

    def _fit_batch(
        self,
        state: TrainingRunState[CatBoostClassifier],
        batch_data: PreparedDataset,
        validation_data: PreparedDataset,
        model_config: Mapping[str, Any],
    ) -> CatBoostClassifier:
        """Fit one batch, warm-started from the preceding CatBoost model."""
        training_weight = (
            batch_data.sample_weight if state.config.use_sample_weight else None
        )
        validation_weight = (
            validation_data.sample_weight if state.config.use_sample_weight else None
        )
        train_pool = Pool(
            batch_data.features,
            batch_data.target,
            weight=training_weight,
            cat_features=list(self._category_schema) or None,
        )
        validation_pool = Pool(
            validation_data.features,
            validation_data.target,
            weight=validation_weight,
            cat_features=list(self._category_schema) or None,
        )
        model = CatBoostClassifier(**model_config)
        model.fit(
            train_pool,
            eval_set=validation_pool,
            init_model=state.previous_model,
            use_best_model=False,
        )
        return model
