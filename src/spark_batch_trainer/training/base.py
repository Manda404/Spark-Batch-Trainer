"""Shared, backend-neutral batch-training workflow."""

from abc import ABC, abstractmethod
from logging import getLogger
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
        self._model: Optional[ModelT] = None

    def _reset_run_history(self) -> None:
        """Clear diagnostics and learned categories before a new fit run."""
        self._global_train_loss.clear()
        self._global_valid_loss.clear()
        self._global_iterations.clear()
        self._lr_schedulers.clear()
        self._category_schema.clear()

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

    @abstractmethod
    def fit(
        self,
        train_dataframe: Optional[SparkDataFrame],
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
        **kwargs: Any,
    ) -> None:
        """Train a model on Spark DataFrames in batches."""
        raise NotImplementedError

    def _extract_metric_history(
        self, model: Any, framework: FrameworkName
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
            metric_name = next(iter(results[train_key]))
            train_scores = results[train_key][metric_name]
            valid_scores = results[valid_key][metric_name]
        except (KeyError, StopIteration) as error:
            raise ValueError(
                f"Incomplete evaluation results from {framework}"
            ) from error
        if not train_scores or not valid_scores:
            raise ValueError(f"Empty evaluation history returned by {framework}")
        return train_scores, valid_scores, metric_name

    def _validate_inputs(
        self,
        train_dataframe: Optional[SparkDataFrame],
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
        num_batches: int,
    ) -> tuple[SparkDataFrame, SparkDataFrame]:
        """Validate the input contract and narrow optional DataFrames."""
        if train_dataframe is None or valid_dataframe is None:
            raise ValueError("train dataframe and validation dataframe cannot be None")
        if num_batches < 1:
            raise ValueError("the number of batches must be >= 1")
        if target_column not in train_dataframe.columns:
            raise ValueError(f"train dataframe must contain '{target_column}' column")
        if target_column not in valid_dataframe.columns:
            raise ValueError(f"valid dataframe must contain '{target_column}' column")
        return train_dataframe, valid_dataframe

    def _resolve_learning_rate(
        self,
        config: Optional[Mapping[str, Any]],
        batch_id: int,
        default_lr: float = 0.1,
    ) -> float:
        """Return the configured or exponentially decayed batch learning rate."""
        if config is None:
            return default_lr

        initial = float(config.get("initial_lr", 0.1))
        decay = float(config.get("decay_rate", 0.95))
        minimum = float(config.get("min_lr", 1e-4))
        if initial <= 0:
            raise ValueError("initial_lr must be > 0")
        if not 0 < decay <= 1:
            raise ValueError("decay_rate must be in the interval (0, 1]")
        if batch_id < 1:
            raise ValueError("batch_id must be >= 1")
        if minimum <= 0:
            raise ValueError("min_lr must be > 0")

        learning_rate = max(minimum, initial * decay ** (batch_id - 1))
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
            model, framework
        )
        train_score = train_scores[-1]
        valid_score = valid_scores[-1]
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

    def _finalize_run(
        self,
        state: TrainingRunState[ModelT],
        *,
        model_name: str,
    ) -> ModelT:
        """Select the best model and render optional diagnostics."""
        final_model = state.best_model or state.previous_model
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
