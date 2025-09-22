from numpy import cumsum
from pyspark.sql import Window
from abc import ABC, abstractmethod
from pandas import DataFrame as PandasDataFrame
from pyspark.sql.functions import col, ntile, rand
from pyspark.sql import DataFrame as SparkDataFrame
from typing import List, Generator, Optional, Tuple, Any, Dict
from spark_batch_trainer.logger.logger import setup_logger
from matplotlib.pyplot import (
    figure,
    plot,
    gca,
    title,
    xlabel,
    ylabel,
    legend,
    grid,
    show,
)


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

      - ``_create_and_apply_batches``: Stratified splitting into batches using Spark.
      - ``_apply_pandas_processing_to_generator``: Convert Spark batches to pandas DataFrames (generator).
      - ``_apply_pandas_processing``: Convert a full Spark DataFrame into pandas.

    - **Learning curve visualization**

      - ``_plot_learning_curve``: Plot flattened training/validation loss across batches.

    - **Data preprocessing**

      - ``_convert_object_to_category_dtype``: Convert object-type columns in pandas DataFrame to categorical.

    - **Learning rate scheduling**

      - ``_exponential_lr_schedule``: Compute exponentially decayed learning rates per batch.
      - ``_plot_lr_schedule``: Plot learning rate evolution across batches.

    - **Training metrics extraction**

      - ``_get_training_metrics``: Extract per-iteration training and validation metrics
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
        self._logger = setup_logger(__name__)

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

    def _create_and_apply_batches(
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
        # Retrieve number of batches from kwargs:
        num_batches = int(kwargs.get("num_batches", 10))

        # Validate the number of batches
        if num_batches <= 0:
            raise ValueError("num_batches must be >= 1")

        # Validate that the target column exists in the dataset
        if target_column not in dataframe.columns:
            raise ValueError(f"DataFrame must contain '{target_column}' column")

        self._logger.info("Creating stratified batches...")

        # Define a window that partitions by target class and assigns a random order
        # (ensures balanced batches while preserving class distribution)
        w = Window.partitionBy(target_column).orderBy(rand(seed=42))

        # Assign each row to a batch using ntile (1..K), then shift to 0..K-1
        df = dataframe.withColumn("batch_id", ntile(num_batches).over(w) - 1)

        self._logger.info(
            "Successfully created %d stratified batches by target column '%s'",
            num_batches,
            target_column,
        )

        return df

    def _apply_pandas_processing_to_generator(
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
        self._logger.info("Starting to process DataFrame in batches...")

        # Create and assign batch IDs to the Spark DataFrame
        dataframe = self._create_and_apply_batches(
            dataframe, batch_column, num_batches=num_batches
        )
        self._logger.info("Batches successfully created and applied to DataFrame.")

        # Sequentially process each batch
        for batch_id in range(num_batches):
            # Filter rows belonging to the current batch
            self._logger.info(f"Filtering batch {batch_id + 1}/{num_batches}")
            batch_dataframe = dataframe.filter(col("batch_id") == batch_id)

            # Convert Spark DataFrame to pandas for local processing
            self._logger.info(
                f"Converting Spark DataFrame to pandas DataFrame for batch {batch_id + 1}/{num_batches}"
            )
            pandas_df = batch_dataframe.toPandas().drop("batch_id", axis=1)

            # Yield the pandas DataFrame for the current batch
            yield pandas_df

    def _apply_pandas_processing(self, dataframe: SparkDataFrame) -> PandasDataFrame:
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
        # Convert the full Spark DataFrame to pandas
        self._logger.info(
            "Converting Spark DataFrame to pandas DataFrame for validation."
        )
        pandas_df: PandasDataFrame = dataframe.toPandas()

        # Log the resulting DataFrame shape for debugging purposes
        self._logger.debug(
            "Pandas DataFrame created with %d rows and %d columns",
            pandas_df.shape[0],
            pandas_df.shape[1],
        )

        # Return the processed DataFrame
        return pandas_df

    def _plot_learning_curve(
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
        # Log the start of the plotting process
        self._logger.info(f"Plotting learning curve of {model_name} model...")

        # Flatten the list of per-batch losses into a single sequence
        flattened_train_loss = [
            loss for batch_curve in global_train_loss for loss in batch_curve
        ]
        flattened_val_loss = [
            loss for batch_curve in global_valid_loss for loss in batch_curve
        ]

        # Compute the cumulative starting index for each batch
        batch_start_indices = cumsum(
            [0] + [len(curve) for curve in global_train_loss[:-1]]
        )

        # Initialize the figure
        figure(figsize=(20, 6))

        # Plot training and validation losses
        plot(flattened_train_loss, label="Train Logloss (flattened)")
        plot(flattened_val_loss, label="Validation Logloss (flattened)")

        # Add vertical lines and labels to indicate batch boundaries
        ax1 = gca()
        epoch = 1
        for idx in range(len(batch_start_indices)):
            batch_num = global_iterations[idx]

            # Draw vertical line at the start of each batch
            ax1.axvline(
                x=batch_start_indices[idx],
                color="red",
                linestyle="--",
                linewidth=0.8,
            )

            # Annotate the plot with epoch and batch information
            ax1.text(
                batch_start_indices[idx],
                ax1.get_ylim()[1] * 0.95,
                f"Epoch {epoch} - Batch {batch_num}",
                ha="center",
                fontsize=10,
                color="gray",
            )

        # Add titles and labels
        title(f"Global Flattened Learning Curve Over All Batches ({model_name})")
        xlabel("Global Iteration (All Batches Concatenated)")
        ylabel(f"{eval_metric}")

        # Add legend and grid for better readability
        legend()
        grid()

        # Display the plot
        show()

        # Log successful plot creation
        self._logger.info("Learning curve plotted successfully.")

    def _convert_object_to_category_dtype(
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
        # Identify candidate categorical features:
        # all columns of type 'object' except the target column
        cat_features = [
            col
            for col in data.select_dtypes(include=["object"]).columns.tolist()
            if col != target_column
        ]

        # If categorical features are found, convert them to pandas 'category' dtype
        if cat_features:
            self._logger.debug(
                f"Converting {len(cat_features)} features to 'category': {cat_features}"
            )
            data[cat_features] = data[cat_features].astype("category")
            is_present = True
        else:
            # No object-type columns to convert
            is_present = False

        # Return updated DataFrame and flag indicating categorical presence
        return data, is_present

    def _exponential_lr_schedule(
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
        >>> trainer._exponential_lr_schedule(0.1, 0.9, 3)
        0.081
        >>> trainer._exponential_lr_schedule(0.001, 0.5, 10, min_lr=0.0005)
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

    def _plot_lr_schedule(
        self,
        name: str,
        lrs: List[float],
        titlename: str = "Learning Rate Schedule",
    ) -> None:
        """
        Plot a learning-rate schedule.

        Parameters
        ----------
        name : str
            Scheduler name (e.g., "ExponentialLR").
        lrs : list of float
            Sequence of learning rates.
        titlename : str, optional
            Plot title (default="Learning Rate Schedule").

        Returns
        -------
        None
            Displays the matplotlib plot.
        """
        # If no learning rates are available, skip plotting
        if not lrs:
            self._logger.warning(
                "Skipping learning-rate schedule plot: empty 'lrs' for scheduler '%s'",
                name,
            )
            return

        # Log information about the scheduler and number of points
        self._logger.info(
            "Plotting learning-rate schedule (scheduler=%s, n_points=%d)",
            name,
            len(lrs),
        )

        # Prepare x-axis as batch indices
        batch_ids = list(range(1, len(lrs) + 1))

        # Plot learning rate evolution
        figure(figsize=(16, 4))
        plot(batch_ids, lrs, marker="o", linestyle="-", label=name)

        # Add plot title, labels, and legend
        title(f"{titlename} - {name}")
        xlabel("Batch")
        ylabel("Learning Rate")
        legend()
        grid(True)

        # Show the plot
        show()

    def _get_training_metrics(
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