"""In-place pandas memory footprint reduction."""

from numpy import iinfo
from pandas import DataFrame
from spark_batch_trainer.logging import configure_logger


class PandasMemoryOptimizer:
    """Utility class to optimize the memory usage of a Pandas DataFrame.

    Optimizations include:

    - Converting ``object`` columns to ``category`` when appropriate.
    - Downcasting floats (``float64`` → ``float32``).
    - Downcasting integers (``int64`` → ``int32`` when possible).

    Example:
        >>> import pandas as pd
        >>> from spark_batch_trainer.data import PandasMemoryOptimizer
        >>> dataframe = pd.DataFrame({
        ...     "category_col": ["a", "b", "a", "c", "b"],
        ...     "int_col": [1, 2, 3, 4, 5],
        ...     "float_col": [1.0, 2.5, 3.2, 4.8, 5.1]
        ... })
        >>> optimizer = PandasMemoryOptimizer(enable_logging=False)
        >>> optimized = optimizer.optimize(dataframe)
        >>> optimized.dtypes
        category_col    category
        int_col           int32
        float_col       float32
        dtype: object
    """

    def __init__(self, enable_logging: bool = True):
        """Initialize the memory optimizer.

        Args:
            enable_logging: If ``True``, logging is enabled to display
                optimization steps and memory usage statistics. Defaults
                to ``True``.
        """
        self.enable_logging = enable_logging
        self._logger = configure_logger(__name__)

    def optimize(self, dataframe: DataFrame) -> DataFrame:
        """Optimize the memory usage of a Pandas DataFrame, in place.

        The optimization includes:

        - Converting object columns to category when the ratio of unique values
          to total values is less than 0.5.
        - Downcasting float64 columns to float32.
        - Downcasting int64 columns to int32 if values fit within bounds.

        Args:
            dataframe: Input DataFrame to optimize, modified in place.

        Returns:
            pandas.DataFrame: The same DataFrame, with reduced memory footprint.
        """
        if self.enable_logging:
            self._logger.info("Starting pandas memory optimization")
        initial_memory = dataframe.memory_usage(deep=True).sum() / 1024**2
        initial_shape = dataframe.shape

        for column_name in dataframe.columns:
            col_type = dataframe[column_name].dtype

            # Object -> Category (if not already category)
            if col_type == "object":
                num_unique_values = dataframe[column_name].nunique(dropna=False)
                num_total_values = len(dataframe[column_name])
                if (
                    num_total_values > 0
                    and (num_unique_values / num_total_values) < 0.5
                ):
                    dataframe[column_name] = dataframe[column_name].astype("category")
                    self._logger.debug(f"Column '{column_name}' converted to 'category'.")

            # Float64 -> Float32
            elif col_type == "float64":
                dataframe[column_name] = dataframe[column_name].astype("float32")
                self._logger.debug(f"Column '{column_name}' converted to 'float32'.")

            # Int64 -> Int32 (if within bounds)
            elif col_type == "int64":
                min_val, max_val = dataframe[column_name].min(), dataframe[column_name].max()
                if iinfo("int32").min <= min_val and max_val <= iinfo("int32").max:
                    dataframe[column_name] = dataframe[column_name].astype("int32")
                    self._logger.debug(f"Column '{column_name}' converted to 'int32'.")

        optimized_memory = dataframe.memory_usage(deep=True).sum() / 1024**2
        optimized_shape = dataframe.shape

        if self.enable_logging:
            self._logger.info(f"Shape before optimization: {initial_shape}")
            self._logger.info(f"Shape after optimization:  {optimized_shape}")
            self._logger.info(f"Memory before: {initial_memory:.2f} MB")
            self._logger.info(f"Memory after:  {optimized_memory:.2f} MB")
            self._logger.info(
                f"Memory reduced by: {initial_memory - optimized_memory:.2f} MB"
            )
            self._logger.info("Pandas memory optimization completed")
        return dataframe
