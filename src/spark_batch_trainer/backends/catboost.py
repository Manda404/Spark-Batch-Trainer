"""CatBoost implementation of the shared batch-training workflow."""

from typing import Any, Mapping, Optional

from catboost import CatBoostClassifier, Pool
from pyspark.sql import DataFrame as SparkDataFrame

from spark_batch_trainer.data.spark_batching import iter_pandas_batches
from spark_batch_trainer.training.base import BatchTrainer
from spark_batch_trainer.training.config import LearningRateConfig, TrainingConfig
from spark_batch_trainer.training.state import PreparedDataset, TrainingRunState


class CatBoostTrainer(BatchTrainer[CatBoostClassifier]):
    """Train a CatBoost classifier incrementally on stratified Spark batches."""

    def fit(
        self,
        train_dataframe: SparkDataFrame,
        valid_dataframe: SparkDataFrame,
        target_column: str,
        *,
        model_config: Optional[Mapping[str, Any]] = None,
        training_config: Optional[Mapping[str, Any] | TrainingConfig] = None,
        learning_rate_config: Optional[Mapping[str, Any] | LearningRateConfig] = None,
    ) -> None:
        """Train the continuation model through every configured batch."""
        resolved_model_config = dict(model_config or {})
        resolved_model_config.setdefault("allow_writing_files", False)
        state = TrainingRunState[CatBoostClassifier].from_mapping(
            training_config,
            default_eval_metric=str(
                resolved_model_config.get(
                    "eval_metric",
                    resolved_model_config.get("loss_function", "Logloss"),
                )
            ),
        )

        self._reset_run_history()
        if learning_rate_config is not None:
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

        def fit_batch(
            _batch_number: int, batch_data: PreparedDataset
        ) -> CatBoostClassifier:
            return self._fit_batch(
                state, batch_data, validation_data, resolved_model_config
            )

        self._model = self._run_batches(
            batches,
            state,
            target_column,
            framework="catboost",
            model_name="CatBoost",
            fit_batch=fit_batch,
        )

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
