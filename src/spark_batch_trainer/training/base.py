"""Shared, backend-neutral batch-training workflow."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from logging import getLogger
from math import isfinite
from typing import Any, Generic, Literal, Mapping, Optional, TypeVar, cast

from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame

from spark_batch_trainer.data.pandas_memory import downcast_numeric_features
from spark_batch_trainer.data.pandas_preparation import (
    CategorySchema,
    convert_categories,
    learn_category_schema,
)
from spark_batch_trainer.data.sample_weighting import calculate_sample_weights
from spark_batch_trainer.training.config import LearningRateConfig, TrainingConfig
from spark_batch_trainer.training.early_stopping import observe_early_stopping
from spark_batch_trainer.training.history import TrainingHistory
from spark_batch_trainer.training.learning_curves import (
    plot_learning_rates,
    plot_metrics,
)
from spark_batch_trainer.training.state import PreparedDataset, TrainingRunState

ModelT = TypeVar("ModelT")
FrameworkName = Literal["lightgbm", "xgboost", "catboost"]


class BatchTrainer(ABC, Generic[ModelT]):
    """Own the workflow shared by every concrete model backend."""

    def __init__(self) -> None:
        self._logger = getLogger(__name__)
        self._global_train_loss: list[list[float]] = []
        self._global_valid_loss: list[list[float]] = []
        self._global_iterations: list[int] = []
        self._lr_schedulers: list[float] = []
        self._category_schema: CategorySchema = {}
        self._feature_columns: tuple[str, ...] = ()
        self._model: Optional[ModelT] = None

    def _reset_run_history(self) -> None:
        """Clear all public artifacts before a new fit run."""
        self._global_train_loss.clear()
        self._global_valid_loss.clear()
        self._global_iterations.clear()
        self._lr_schedulers.clear()
        self._category_schema.clear()
        self._feature_columns = ()
        self._model = None

    def get_training_history(self) -> TrainingHistory:
        """Return an immutable snapshot of metrics from the latest fit run."""
        return TrainingHistory(
            tuple(tuple(values) for values in self._global_train_loss),
            tuple(tuple(values) for values in self._global_valid_loss),
            tuple(self._global_iterations),
            tuple(self._lr_schedulers),
        )

    def get_trained_model(self) -> Optional[ModelT]:
        """Return the best model from the latest completed fit run."""
        return self._model

    def prepare_features(self, dataframe: PandasDataFrame) -> PandasDataFrame:
        """Apply the fitted feature order, categorical schema, and downcasting.

        Args:
            dataframe: In-memory features to prepare for bounded inference. A
                target column may be present and is ignored.

        Returns:
            A prepared copy with exactly the feature columns seen during fit.

        Raises:
            RuntimeError: If the trainer has not completed a fit run.
            ValueError: If a required feature is missing.
        """
        if self._model is None or not self._feature_columns:
            raise RuntimeError("the trainer must be fitted before preparing features")
        missing = [
            name for name in self._feature_columns if name not in dataframe.columns
        ]
        if missing:
            raise ValueError(f"missing inference feature columns: {missing}")
        features = dataframe.loc[:, self._feature_columns].copy()
        convert_categories(features, self._category_schema)
        return downcast_numeric_features(features)

    @abstractmethod
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
        """Train a model on Spark DataFrames in batches."""
        raise NotImplementedError

    def _extract_metric_history(
        self,
        model: Any,
        framework: FrameworkName,
        monitor_metric: Optional[str] = None,
    ) -> tuple[list[float], list[float], str]:
        """Extract train and validation history from a backend model."""
        if framework == "lightgbm":
            results = cast(dict[str, dict[str, list[float]]], model.evals_result_)
            train_key, valid_key = "train", "valid"
        elif framework == "xgboost":
            results = cast(dict[str, dict[str, list[float]]], model.evals_result())
            train_key, valid_key = "validation_0", "validation_1"
        else:
            results = cast(dict[str, dict[str, list[float]]], model.get_evals_result())
            train_key, valid_key = "learn", "validation"

        try:
            train_metrics = results[train_key]
            valid_metrics = results[valid_key]
            common_metrics = [name for name in train_metrics if name in valid_metrics]
            if monitor_metric is None:
                if len(common_metrics) != 1:
                    raise ValueError(
                        f"{framework} returned multiple metrics {common_metrics}; "
                        "set training_config['monitor_metric'] explicitly"
                    )
                metric_name = common_metrics[0]
            else:
                metric_name = next(
                    name
                    for name in common_metrics
                    if name.casefold() == monitor_metric.casefold()
                )
            train_scores = results[train_key][metric_name]
            valid_scores = results[valid_key][metric_name]
        except (KeyError, StopIteration) as error:
            raise ValueError(
                f"Incomplete evaluation results from {framework}"
            ) from error
        if not train_scores or not valid_scores:
            raise ValueError(f"Empty evaluation history returned by {framework}")
        return list(train_scores), list(valid_scores), metric_name

    def _validate_inputs(
        self,
        train_dataframe: SparkDataFrame,
        valid_dataframe: SparkDataFrame,
        target_column: str,
        num_batches: int,
    ) -> tuple[SparkDataFrame, SparkDataFrame]:
        """Validate schema, labels, and batch viability before training."""
        if train_dataframe is None or valid_dataframe is None:
            raise ValueError("train dataframe and validation dataframe cannot be None")
        if num_batches < 1:
            raise ValueError("the number of batches must be >= 1")
        if target_column not in train_dataframe.columns:
            raise ValueError(f"train dataframe must contain '{target_column}' column")
        if target_column not in valid_dataframe.columns:
            raise ValueError(f"valid dataframe must contain '{target_column}' column")

        train_features = [
            name for name in train_dataframe.columns if name != target_column
        ]
        valid_features = [
            name for name in valid_dataframe.columns if name != target_column
        ]
        if not train_features:
            raise ValueError("at least one feature column is required")
        if train_features != valid_features:
            raise ValueError(
                "train and validation feature columns must match in the same order"
            )

        train_types = {
            field.name: field.dataType.simpleString()
            for field in train_dataframe.schema.fields
        }
        valid_types = {
            field.name: field.dataType.simpleString()
            for field in valid_dataframe.schema.fields
        }
        mismatched_types = [
            name
            for name in train_dataframe.columns
            if train_types[name] != valid_types[name]
        ]
        if mismatched_types:
            raise ValueError(
                f"train and validation column types differ: {mismatched_types}"
            )

        train_label_counts = {
            row[target_column]: row["count"]
            for row in train_dataframe.groupBy(target_column).count().collect()
        }
        valid_labels = {
            row[target_column]
            for row in valid_dataframe.select(target_column).distinct().collect()
        }
        if not train_label_counts:
            raise ValueError("train dataframe must not be empty")
        if not valid_labels:
            raise ValueError("validation dataframe must not be empty")
        if None in train_label_counts or None in valid_labels:
            raise ValueError("target values must not contain nulls")
        if len(train_label_counts) < 2:
            raise ValueError("training target must contain at least two classes")
        unknown_valid_labels = valid_labels - set(train_label_counts)
        if unknown_valid_labels:
            raise ValueError(
                f"validation contains labels absent from training: "
                f"{sorted(unknown_valid_labels, key=str)}"
            )
        smallest_class = min(train_label_counts.values())
        if smallest_class < num_batches:
            raise ValueError(
                f"num_batches={num_batches} exceeds the smallest training class "
                f"size ({smallest_class})"
            )
        self._feature_columns = tuple(train_features)
        return train_dataframe, valid_dataframe

    def _resolve_learning_rate(
        self,
        config: Optional[Mapping[str, Any] | LearningRateConfig],
        batch_id: int,
        default_lr: float = 0.1,
    ) -> float:
        """Return the configured or exponentially decayed batch learning rate."""
        if config is None:
            return default_lr

        resolved = LearningRateConfig.from_mapping(config)
        if batch_id < 1:
            raise ValueError("batch_id must be >= 1")

        learning_rate = max(
            resolved.min_lr,
            resolved.initial_lr * resolved.decay_rate ** (batch_id - 1),
        )
        self._lr_schedulers.append(learning_rate)
        return learning_rate

    def _prepare_batch(
        self,
        pandas_batch: PandasDataFrame,
        target_column: str,
        *,
        use_sample_weight: bool,
    ) -> PreparedDataset:
        """Prepare one collected training batch for a native model backend."""
        sample_weight = (
            calculate_sample_weights(pandas_batch[target_column])
            if use_sample_weight
            else None
        )
        convert_categories(pandas_batch, self._category_schema)
        return PreparedDataset(
            features=downcast_numeric_features(
                pandas_batch.drop(columns=[target_column])
            ),
            target=pandas_batch[target_column],
            sample_weight=sample_weight,
        )

    def _prepare_validation(
        self,
        train_dataframe: SparkDataFrame,
        valid_dataframe: SparkDataFrame,
        target_column: str,
        *,
        use_sample_weight: bool,
    ) -> PreparedDataset:
        """Learn categories, then collect and prepare validation data once."""
        self._category_schema = learn_category_schema(
            train_dataframe, valid_dataframe, target_column
        )
        self._logger.info("Collecting the complete validation DataFrame on the driver")
        validation = cast(PandasDataFrame, valid_dataframe.toPandas())
        convert_categories(validation, self._category_schema)
        sample_weight = (
            calculate_sample_weights(validation[target_column])
            if use_sample_weight
            else None
        )
        return PreparedDataset(
            features=downcast_numeric_features(
                validation.drop(columns=[target_column])
            ),
            target=validation[target_column],
            sample_weight=sample_weight,
        )

    def _evaluate_model(
        self,
        model: ModelT,
        state: TrainingRunState[ModelT],
        batch_number: int,
        framework: FrameworkName,
    ) -> bool:
        """Record metrics and update best-model and early-stopping state."""
        train_scores, valid_scores, metric_name = self._extract_metric_history(
            model, framework, state.config.monitor_metric
        )
        train_score = train_scores[-1]
        valid_score = valid_scores[-1]
        if not isfinite(train_score) or not isfinite(valid_score):
            raise ValueError(
                f"Non-finite {metric_name} returned at batch {batch_number}: "
                f"train={train_score}, validation={valid_score}"
            )
        self._global_train_loss.append(train_scores)
        self._global_valid_loss.append(valid_scores)
        self._global_iterations.append(batch_number)
        self._logger.info(
            "Batch %d - %s | Train: %.5f | Valid: %.5f",
            batch_number,
            metric_name,
            train_score,
            valid_score,
        )

        state.previous_model = model
        state.observed_metric = metric_name
        decision = observe_early_stopping(
            current_score=valid_score,
            best_score=state.best_valid_score,
            metric_name=metric_name,
            patience_counter=state.patience_counter,
            max_patience=state.config.max_patience,
            mode=state.config.metric_mode,
            min_delta=state.config.min_delta,
            logger=self._logger,
        )
        state.patience_counter = decision.patience_counter
        if decision.improved:
            state.best_valid_score = decision.best_score
            state.best_model = model
            message = "Initial best" if decision.improvement is None else "New best"
            self._logger.info("%s %s: %.5f", message, metric_name, valid_score)
        else:
            self._logger.info(
                "No improvement - Patience: %d/%d",
                state.patience_counter,
                state.config.max_patience,
            )
        return decision.should_stop

    def _run_batches(
        self,
        batches: Iterable[PandasDataFrame],
        state: TrainingRunState[ModelT],
        target_column: str,
        *,
        framework: FrameworkName,
        model_name: str,
        fit_batch: Callable[[int, PreparedDataset], ModelT],
    ) -> ModelT:
        """Run the backend-neutral preparation, fitting, and evaluation loop."""
        self._logger.info(
            "Starting %s training with %d batches",
            model_name,
            state.config.num_batches,
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
                model = fit_batch(batch_number, batch_data)
                if self._evaluate_model(model, state, batch_number, framework):
                    self._logger.info("Global early stopping triggered")
                    break
        except Exception:
            self._logger.exception(
                "%s training failed at batch %d", model_name, batch_number
            )
            self._reset_run_history()
            raise
        return self._finalize_run(state, model_name=model_name)

    def _finalize_run(
        self,
        state: TrainingRunState[ModelT],
        *,
        model_name: str,
    ) -> ModelT:
        """Select the best model and render optional diagnostics."""
        final_model = (
            state.best_model if state.best_model is not None else state.previous_model
        )
        if final_model is None:
            raise RuntimeError("No model was successfully trained")

        if state.config.show_learning_curve:
            if self._lr_schedulers:
                plot_learning_rates("Exponential Decay", self._lr_schedulers)
            plot_metrics(
                self._global_train_loss,
                self._global_valid_loss,
                self._global_iterations,
                model_name,
                state.observed_metric or state.eval_metric,
            )

        self._logger.info(
            "%s training completed after %d batches",
            model_name,
            len(self._global_iterations),
        )
        return final_model
