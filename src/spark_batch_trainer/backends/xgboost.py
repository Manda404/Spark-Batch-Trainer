"""XGBoost implementation of the shared batch-training workflow."""

from typing import Any, Mapping, Optional

from pyspark.sql import DataFrame as SparkDataFrame
from xgboost import XGBClassifier

from spark_batch_trainer.data.spark_batching import iter_pandas_batches
from spark_batch_trainer.training.base import BatchTrainer
from spark_batch_trainer.training.state import PreparedDataset, TrainingRunState


class XGBoostTrainer(BatchTrainer[XGBClassifier]):
    """Train an XGBoost classifier incrementally on stratified Spark batches."""

    def fit(
        self,
        train_dataframe: Optional[SparkDataFrame],
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
        **kwargs: Any,
    ) -> None:
        """Train the model and retain the best validation checkpoint.

        Keyword Args:
            model_config: Parameters passed to :class:`XGBClassifier`.
            training_config: Shared batching, weighting, plotting, and early-
                stopping options.
            learning_rate_config: Optional exponential-decay configuration.
        """
        model_config = dict(kwargs.get("model_config") or {})
        training_values: Optional[Mapping[str, Any]] = kwargs.get("training_config")
        learning_rate_config: Optional[Mapping[str, Any]] = kwargs.get(
            "learning_rate_config"
        )
        state = TrainingRunState[XGBClassifier].from_mapping(
            training_values,
            default_eval_metric=str(model_config.get("eval_metric", "logloss")),
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
        if self._category_schema:
            model_config["enable_categorical"] = True

        batches = iter_pandas_batches(
            train_dataframe,
            target_column,
            state.config.num_batches,
            self._logger,
        )
        self._logger.info(
            "Starting XGBoost training with %d batches", state.config.num_batches
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
                model = XGBClassifier(
                    **{**model_config, "learning_rate": learning_rate}
                )
                self._fit_batch(model, state, batch_data, validation_data)
                if self._evaluate_model(
                    model, state, batch_number, framework="xgboost"
                ):
                    self._logger.info("Global early stopping triggered")
                    break
        except Exception:
            self._logger.exception("XGBoost training failed at batch %d", batch_number)
            raise

        self._model = self._finalize_run(state, model_name="XGBoost")

    def _fit_batch(
        self,
        model: XGBClassifier,
        state: TrainingRunState[XGBClassifier],
        batch_data: PreparedDataset,
        validation_data: PreparedDataset,
    ) -> None:
        """Fit one batch, continuing from the preceding booster when present."""
        previous_booster = (
            state.previous_model.get_booster()
            if state.previous_model is not None
            else None
        )
        evaluation_weights = None
        if state.config.use_sample_weight:
            evaluation_weights = [
                batch_data.sample_weight,
                validation_data.sample_weight,
            ]

        model.fit(
            batch_data.features,
            batch_data.target,
            eval_set=[
                (batch_data.features, batch_data.target),
                (validation_data.features, validation_data.target),
            ],
            xgb_model=previous_booster,
            sample_weight=(
                batch_data.sample_weight if state.config.use_sample_weight else None
            ),
            sample_weight_eval_set=evaluation_weights,
        )
