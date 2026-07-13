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
    """
    Abstract base class for batch trainers.

    This class defines the interface and provides common utilities
    for implementing batch-wise training with Spark and pandas.
    It is designed to be subclassed by concrete trainers
    (e.g., XGBoost, CatBoost, LightGBM).

    Main Features
    -------------
    - **Training interface**

      - ``fit``: Abstract method to train a model in batches.
      - ``get_trained_model``: Abstract method to retrieve the trained model.

    - **Batch creation & processing**

      - ``_assign_batches``: Stratified splitting into batches using Spark.
      - ``_iter_training_batches``: Convert Spark batches to pandas DataFrames (generator).
      - ``_collect_validation_data``: Convert a full Spark DataFrame into pandas.

    - **Learning curve visualization**

      - ``_plot_metric_history``: Plot flattened training/validation loss across batches.

    - **Data preprocessing**

      - ``_convert_categorical_features``: Convert object-type columns in pandas DataFrame to categorical.

    - **Learning rate scheduling**

      - ``_calculate_scheduled_learning_rate``: Compute exponentially decayed learning rates per batch.
      - ``_plot_learning_rate_history``: Plot learning rate evolution across batches.

    - **Training metrics extraction**

      - ``_extract_metric_history``: Extract per-iteration training and validation metrics
        from LightGBM, XGBoost, or CatBoost models.

    Attributes
    ----------
    _logger : Logger
        Logger instance used for logging training, preprocessing, and plotting information.

    Notes
    -----
    - This is an abstract class and cannot be instantiated directly.
    - Subclasses must implement at least ``fit`` and ``get_trained_model``.
    - Designed for scalable ML workflows combining Spark preprocessing and batch-wise training.
    """
    def __init__(self):
        """
        """
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
        """
        Abstract method to train a model on Spark DataFrames in batches.

        Parameters
        ----------
        train_dataframe : SparkDataFrame or None
            Training dataset.
        valid_dataframe : SparkDataFrame or None
            Validation dataset.
        target_column : str
            Name of the target column.
        **kwargs : dict
            Additional training configurations.
        """
        pass

    @abstractmethod
    def get_trained_model(self) -> Any:
        """
        Abstract method to retrieve the trained model instance.

        Returns
        -------
        Any
            The trained model object (depends on implementation).
        """
        pass

    def _assign_batches(
        self, dataframe: SparkDataFrame, target_column: str, **kwargs
    ) -> SparkDataFrame:
        """
        Create stratified training batches per target class using Spark's ``ntile``.

        This ensures that each batch contains approximately the same distribution
        of the target classes, which is useful for imbalanced datasets.

        Parameters
        ----------
        dataframe : SparkDataFrame
            Input dataset.
        target_column : str
            Target column used for stratified batching.
        **kwargs : dict
            Additional arguments, such as:
            - ``num_batches`` (int, default=10): Number of batches to create.

        Returns
        -------
        SparkDataFrame
            Input DataFrame with an additional ``batch_id`` column
            indicating the assigned batch index.

        Raises
        ------
        ValueError
            If ``num_batches`` <= 0 or if ``target_column`` is missing.

        Notes
        -----
        - Uses ``ntile`` to split each class into evenly sized batches.
        - A fixed random seed (42) ensures reproducibility.
        - Batch indices start at 0 and go up to ``num_batches - 1``.
        """
        num_batches = int(kwargs.get("num_batches", 10))
        return self._batcher.assign_batches(dataframe, target_column, num_batches)

    def _iter_training_batches(
        self,
        dataframe: SparkDataFrame,
        batch_column: str,
        num_batches: int,
    ) -> Generator[PandasDataFrame, None, None]:
        """
        Convert Spark DataFrame batches into pandas DataFrames, yielding them sequentially.

        Parameters
        ----------
        dataframe : SparkDataFrame
            Input dataset.
        batch_column : str
            Column used for batch stratification.
        num_batches : int
            Number of batches.

        Yields
        ------
        PandasDataFrame
            Pandas DataFrame for each batch.
        """
        yield from self._batcher.iter_pandas_batches(
            dataframe, batch_column, num_batches
        )

    def _collect_validation_data(self, dataframe: SparkDataFrame) -> PandasDataFrame:
        """
        Convert a Spark DataFrame into a pandas DataFrame.

        Parameters
        ----------
        dataframe : SparkDataFrame
            Input dataset.

        Returns
        -------
        PandasDataFrame
            Equivalent pandas DataFrame.
        """
        return self._data_preparer.collect(dataframe, purpose="validation")

    def _plot_metric_history(
        self,
        global_train_loss: List[List[float]],
        global_valid_loss: List[List[float]],
        global_iterations: List[int],
        model_name: str = "CatBoost",
        eval_metric: str = "Logloss",
    ) -> None:
        """
        Plot the global flattened learning curve over all batches.

        Parameters
        ----------
        global_train_loss : list of list of float
            Training loss per batch.
        global_valid_loss : list of list of float
            Validation loss per batch.
        global_iterations : list of int
            Batch indices corresponding to losses.
        model_name : str, optional
            Model name for plot title (default="CatBoost").
        eval_metric : str, optional
            Evaluation metric name for axis label (default="Logloss").

        Returns
        -------
        None
            Displays the matplotlib plot.
        """
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
        """
        Convert object columns in a pandas DataFrame to categorical dtype.

        Parameters
        ----------
        data : PandasDataFrame
            Input dataset.
        target_column : str, optional
            Target column name (excluded from conversion).

        Returns
        -------
        data : PandasDataFrame
            Updated DataFrame with categorical features converted.
        bool
            Whether categorical features were detected and converted.
        """
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
        """
        Compute exponentially decayed learning rate, bounded by ``min_lr``.

        Formula
        -------
        lr_b = max(min_lr, initial_lr * decay_rate ** (batch_id - 1))

        Parameters
        ----------
        initial_lr : float
            Initial learning rate (> 0).
        decay_rate : float
            Multiplicative decay factor (0 < decay_rate <= 1).
        batch_id : int
            Current batch index (1-indexed).
        min_lr : float, optional
            Minimum learning rate (default=1e-4).

        Returns
        -------
        float
            Learning rate for the current batch.

        Raises
        ------
        ValueError
            If parameters are invalid.

        Examples
        --------
        >>> trainer._calculate_scheduled_learning_rate(0.1, 0.9, 3)
        0.081
        >>> trainer._calculate_scheduled_learning_rate(0.001, 0.5, 10, min_lr=0.0005)
        0.0005
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
        """
        Plot a learning-rate schedule.

        Parameters
        ----------
        name : str
            Scheduler name (e.g., "ExponentialLR").
        learning_rates : list of float
            Sequence of learning rates.
        title : str, optional
            Plot title (default="Learning Rate Schedule").

        Returns
        -------
        None
            Displays the matplotlib plot.
        """
        self._plotter.plot_learning_rates(name, learning_rates)

    def _extract_metric_history(
        self, model: Any, framework: str = "lightgbm"
    ) -> Tuple[List[float], List[float], str]:
        """
        Extract per-iteration training and validation metrics.

        Supports LightGBM, XGBoost, and CatBoost.

        Parameters
        ----------
        model : object
            Trained model instance. Supported types:
            - lightgbm.LGBMClassifier
            - xgboost.XGBClassifier
            - catboost.CatBoostClassifier
        framework : str, optional
            Framework name, one of {"lightgbm", "xgboost", "catboost"}.

        Returns
        -------
        train_metric : list of float
            Training metric values per iteration.
        valid_metric : list of float
            Validation metric values per iteration.
        metric_name : str
            Name of the evaluation metric (e.g., "logloss").

        Raises
        ------
        ValueError
            If the framework is unsupported or no eval results are available.
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
