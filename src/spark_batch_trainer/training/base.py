"""Shared training workflow for all model backends."""

from abc import ABC, abstractmethod
from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame
from typing import List, Generator, Optional, Tuple, Any, Dict
from spark_batch_trainer.logging import configure_logger
from spark_batch_trainer.data import PandasDataPreparer, StratifiedSparkBatcher
from spark_batch_trainer.training.history import TrainingHistory
from spark_batch_trainer.training.learning_curves import LearningCurvePlotter


class BatchTrainer(ABC):
    """Shared batching, preprocessing, and plotting utilities for batch trainers.

    Subclassed by the concrete XGBoost, CatBoost, and LightGBM trainers,
    which implement ``fit`` and ``get_trained_model``.
    """

    def __init__(self):
        """Wire up the shared batching, preparation, and plotting collaborators."""
        self._logger = configure_logger(__name__)
        self._batcher = StratifiedSparkBatcher(self._logger)
        self._data_preparer = PandasDataPreparer(self._logger)
        self._plotter = LearningCurvePlotter(self._logger)

    def _reset_run_history(self) -> None:
        """Clear mutable diagnostics before starting a new fit run."""
        self._global_train_loss.clear()
        self._global_valid_loss.clear()
        self._global_iterations.clear()
        if hasattr(self, "_lr_schedulers"):
            self._lr_schedulers.clear()
        self._categorical_features = None
        self._known_labels = None
        self._data_preparer.reset_category_schema()

    def _fit_category_schema(
        self,
        train_dataframe: SparkDataFrame,
        valid_dataframe: SparkDataFrame,
        target_column: str,
    ) -> None:
        """Learn one categorical mapping reused by validation and all batches."""
        self._categorical_features = self._data_preparer.fit_category_schema(
            train_dataframe,
            valid_dataframe,
            target_column,
        )

    def _fit_known_labels(
        self,
        train_dataframe: SparkDataFrame,
        valid_dataframe: SparkDataFrame,
        target_column: str,
    ) -> None:
        """Learn the global target label space once from train and validation data.

        Sample weights are computed per batch; without a shared label space,
        each batch (and the validation set) would derive its own class list
        from whatever labels it happens to contain, making weight scale
        inconsistent across batches and disabling mean-1 normalization (see
        :meth:`~spark_batch_trainer.data.BalancedSampleWeightCalculator.calculate_sample_weights`).
        """
        distinct_rows = (
            train_dataframe.select(target_column)
            .union(valid_dataframe.select(target_column))
            .distinct()
            .collect()
        )
        self._known_labels = sorted(
            (row[target_column] for row in distinct_rows if row[target_column] is not None),
            key=str,
        )

    def get_training_history(self) -> TrainingHistory:
        """Return an immutable snapshot of metrics from the latest fit run."""
        return TrainingHistory.from_lists(
            self._global_train_loss,
            self._global_valid_loss,
            self._global_iterations,
            getattr(self, "_lr_schedulers", []),
        )

    @abstractmethod
    def fit(
        self,
        train_dataframe: Optional[SparkDataFrame],
        valid_dataframe: Optional[SparkDataFrame],
        target_column: str,
        **kwargs,
    ) -> None:
        """Train a model on Spark DataFrames in batches. Implemented by each backend."""
        pass

    @abstractmethod
    def get_trained_model(self) -> Any:
        """Return the trained model instance. Implemented by each backend."""
        pass

    def _assign_batches(
        self, dataframe: SparkDataFrame, target_column: str, **kwargs
    ) -> SparkDataFrame:
        """Assign a stratified batch id per target class using Spark's ``ntile``."""
        num_batches = int(kwargs.get("num_batches", 10))
        return self._batcher.assign_batches(dataframe, target_column, num_batches)

    def _iter_training_batches(
        self,
        dataframe: SparkDataFrame,
        batch_column: str,
        num_batches: int,
    ) -> Generator[PandasDataFrame, None, None]:
        """Yield one collected pandas batch at a time from the Spark dataset."""
        yield from self._batcher.iter_pandas_batches(
            dataframe, batch_column, num_batches
        )

    def _collect_validation_data(self, dataframe: SparkDataFrame) -> PandasDataFrame:
        """Collect the full validation Spark DataFrame onto the driver as pandas."""
        return self._data_preparer.collect(dataframe, purpose="validation")

    def _plot_metric_history(
        self,
        global_train_loss: List[List[float]],
        global_valid_loss: List[List[float]],
        global_iterations: List[int],
        model_name: str = "CatBoost",
        eval_metric: str = "Logloss",
    ) -> None:
        """Plot the global flattened learning curve over all batches."""
        self._plotter.plot_metrics(
            global_train_loss,
            global_valid_loss,
            global_iterations,
            model_name,
            eval_metric,
        )

    def _convert_categorical_features(
        self, data: PandasDataFrame, target_column: str = ""
    ) -> Tuple[PandasDataFrame, bool]:
        """Convert object columns to categorical dtype; report whether any were found."""
        prepared, categorical_features = self._data_preparer.convert_categories(
            data, target_column
        )
        return prepared, bool(categorical_features)

    def _calculate_scheduled_learning_rate(
        self,
        initial_lr: float,
        decay_rate: float,
        batch_id: int,
        min_lr: float = 1e-4,
    ) -> float:
        """Exponentially decay the learning rate: ``max(min_lr, initial_lr * decay_rate**(batch_id-1))``.

        Raises:
            ValueError: If ``initial_lr``, ``decay_rate``, ``batch_id``, or
                ``min_lr`` is out of range.
        """
        # Validate that the initial learning rate is positive
        if initial_lr <= 0:
            raise ValueError("initial_lr must be > 0")

        # Validate that decay_rate is within (0, 1]
        if not (0 < decay_rate <= 1):
            raise ValueError("decay_rate must be in the interval (0, 1]")

        # Ensure batch index starts from 1 (1-indexed, not 0-indexed)
        if batch_id < 1:
            raise ValueError("batch_id must be >= 1 (1-indexed)")

        # Validate that the minimum learning rate is positive
        if min_lr <= 0:
            raise ValueError("min_lr must be > 0")

        # Apply exponential decay formula
        lr = initial_lr * (decay_rate ** (batch_id - 1))

        # Ensure the learning rate never goes below min_lr
        return max(min_lr, lr)

    def _plot_learning_rate_history(
        self,
        name: str,
        learning_rates: List[float],
        title: str = "Learning Rate Schedule",
    ) -> None:
        """Plot a learning-rate schedule."""
        self._plotter.plot_learning_rates(name, learning_rates)

    def _extract_metric_history(
        self, model: Any, framework: str = "lightgbm"
    ) -> Tuple[List[float], List[float], str]:
        """Extract (train_metric, valid_metric, metric_name) from a trained model's eval history.

        Raises:
            ValueError: If ``framework`` is unsupported or no eval results
                are available.
        """
        evals_result: Optional[Dict[str, Dict[str, list[Any]]]] = None

        # Extract evaluation results depending on the framework
        if framework == "lightgbm":
            evals_result = model.evals_result_
            train_key, valid_key = "train", "valid"

        elif framework == "xgboost":
            evals_result = model.evals_result()
            train_key, valid_key = "validation_0", "validation_1"

        elif framework == "catboost":
            evals_result = model.get_evals_result()
            train_key, valid_key = "learn", "validation" #"validation_0", "validation_1"

        else:
            raise ValueError(
                "framework must be one of: 'lightgbm', 'xgboost', 'catboost'."
            )

        # If no evaluation results are available, raise an error
        if evals_result is None:
            raise ValueError("Evaluation results are not available for this model.")

        # Retrieve the first available metric (e.g., 'logloss', 'accuracy')
        first_key = list(evals_result[train_key].keys())[0]
        self._logger.info(f"Tracking evaluation metric: {first_key}")

        # Return per-iteration metrics and the metric name
        return (
            evals_result[train_key][first_key],  # Training metric values
            evals_result[valid_key][first_key],  # Validation metric values
            first_key,  # Metric name
        )
