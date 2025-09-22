from numpy import iinfo
from pandas import DataFrame
from spark_batch_trainer.logger.logger import setup_logger


class MemoryOptimizer:
    """
    Utility class to optimize the memory usage of a Pandas DataFrame.

    Optimizations include:

    - Converting ``object`` columns to ``category`` when appropriate.
    - Downcasting floats (``float64`` → ``float32``).
    - Downcasting integers (``int64`` → ``int32`` when possible).

    Examples
    --------
    >>> import pandas as pd
    >>> from spark_batch_trainer.core.utils import Optimizer
    >>> 
    >>> # Create a sample DataFrame
    >>> df = pd.DataFrame({
    ...     "category_col": ["a", "b", "a", "c", "b"],
    ...     "int_col": [1, 2, 3, 4, 5],
    ...     "float_col": [1.0, 2.5, 3.2, 4.8, 5.1]
    ... })
    >>> 
    >>> optimizer = Optimizer(use_logger=False)
    >>> df_optimized = optimizer.optimize(df)
    >>> 
    >>> df_optimized.dtypes
    category_col    category
    int_col           int32
    float_col       float32
    dtype: object
    """

    def __init__(self, use_logger: bool = True):
        """
        Initialize the memory optimizer.

        Parameters
        ----------
        use_logger : bool, default=True
            If True, logging is enabled to display optimization steps
            and memory usage statistics.
        """
        self.use_logger = use_logger
        self._logger = setup_logger(__name__)

    def optimize(self, df: DataFrame) -> DataFrame:
        """
        Optimize the memory usage of a Pandas DataFrame.

        The optimization includes:
        
        - Converting object columns to category when the ratio of unique values
          to total values is less than 0.5.
        - Downcasting float64 columns to float32.
        - Downcasting int64 columns to int32 if values fit within bounds.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame to optimize.

        Returns
        -------
        pandas.DataFrame
            Optimized DataFrame with reduced memory footprint.

        Notes
        -----
        - The method modifies the DataFrame in place and returns it.
        - Logging information about the optimization process is displayed
          if ``use_logger=True``.
        """
        self._logger.info("=== Starting memory optimization ===")
        initial_memory = df.memory_usage(deep=True).sum() / 1024**2
        initial_shape = df.shape

        for col in df.columns:
            col_type = df[col].dtype

            # Object -> Category (if not already category)
            if col_type == "object":
                num_unique_values = df[col].nunique(dropna=False)
                num_total_values = len(df[col])
                if (
                    num_total_values > 0
                    and (num_unique_values / num_total_values) < 0.5
                ):
                    df[col] = df[col].astype("category")
                    self._logger.debug(f"Column '{col}' converted to 'category'.")

            # Float64 -> Float32
            elif col_type == "float64":
                df[col] = df[col].astype("float32")
                self._logger.debug(f"Column '{col}' converted to 'float32'.")

            # Int64 -> Int32 (if within bounds)
            elif col_type == "int64":
                min_val, max_val = df[col].min(), df[col].max()
                if iinfo("int32").min <= min_val and max_val <= iinfo("int32").max:
                    df[col] = df[col].astype("int32")
                    self._logger.debug(f"Column '{col}' converted to 'int32'.")

        optimized_memory = df.memory_usage(deep=True).sum() / 1024**2
        optimized_shape = df.shape

        # Logs
        if self.use_logger:
            self._logger.info(f"Shape before optimization: {initial_shape}")
            self._logger.info(f"Shape after optimization:  {optimized_shape}")
            self._logger.info(f"Memory before: {initial_memory:.2f} MB")
            self._logger.info(f"Memory after:  {optimized_memory:.2f} MB")
            self._logger.info(
                f"Memory reduced by: {initial_memory - optimized_memory:.2f} MB"
            )

        self._logger.info("=== Memory optimization finished ===")
        return df