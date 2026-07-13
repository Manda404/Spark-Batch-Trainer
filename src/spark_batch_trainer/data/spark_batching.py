"""Deterministic, stratified batch creation for Spark DataFrames."""

from logging import Logger
from typing import Generator

from pandas import DataFrame as PandasDataFrame
from pyspark import StorageLevel
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import Window
from pyspark.sql.functions import col, ntile, rand


class StratifiedSparkBatcher:
    """Assign target-stratified batch identifiers and collect one batch at a time.

    Spark performs assignment and filtering. Model training remains local, so every
    yielded pandas batch must still fit in driver memory.
    """

    _BATCH_COLUMN = "__spark_batch_trainer_batch_id"

    def __init__(self, logger: Logger, seed: int = 42) -> None:
        self._logger = logger
        self._seed = seed

    def assign_batches(
        self,
        dataframe: SparkDataFrame,
        target_column: str,
        num_batches: int,
    ) -> SparkDataFrame:
        """Return a lazy Spark DataFrame with a private zero-based batch id."""
        if num_batches < 1:
            raise ValueError("num_batches must be >= 1")
        if target_column not in dataframe.columns:
            raise ValueError(f"DataFrame must contain '{target_column}' column")
        if self._BATCH_COLUMN in dataframe.columns:
            raise ValueError(
                f"reserved internal column already exists: {self._BATCH_COLUMN}"
            )

        window = Window.partitionBy(target_column).orderBy(rand(seed=self._seed))
        return dataframe.withColumn(
            self._BATCH_COLUMN, ntile(num_batches).over(window) - 1
        )

    def iter_pandas_batches(
        self,
        dataframe: SparkDataFrame,
        target_column: str,
        num_batches: int,
    ) -> Generator[PandasDataFrame, None, None]:
        """Yield pandas batches while materializing Spark assignment only once."""
        assigned = self.assign_batches(
            dataframe, target_column, num_batches
        ).persist(StorageLevel.MEMORY_AND_DISK)
        try:
            # Materialization prevents the expensive window from being recomputed
            # independently for every batch filter.
            total_rows = assigned.count()
            self._logger.info(
                "Prepared %d rows across %d stratified batches",
                total_rows,
                num_batches,
            )
            for batch_id in range(num_batches):
                self._logger.info(
                    "Collecting batch %d/%d on the driver",
                    batch_id + 1,
                    num_batches,
                )
                batch = (
                    assigned.filter(col(self._BATCH_COLUMN) == batch_id)
                    .drop(self._BATCH_COLUMN)
                    .toPandas()
                )
                if batch.empty:
                    self._logger.warning("Batch %d is empty", batch_id + 1)
                yield batch
        finally:
            assigned.unpersist(blocking=False)
