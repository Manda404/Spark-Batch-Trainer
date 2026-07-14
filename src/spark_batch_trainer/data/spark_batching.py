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
        """Store the logger and the seed used to shuffle rows within each class.

        Args:
            logger: Logger used to report batching progress.
            seed: Random seed controlling the row shuffle performed before
                batch assignment. Fixed by default for reproducible batch
                contents. Defaults to 42.
        """
        self._logger = logger
        self._seed = seed

    def assign_batches(
        self,
        dataframe: SparkDataFrame,
        target_column: str,
        num_batches: int,
    ) -> SparkDataFrame:
        """Return a lazy Spark DataFrame with a private zero-based batch id.

        Args:
            dataframe: Input dataset. Must contain ``target_column`` and
                must not already contain the reserved internal batch-id column.
            target_column: Column used to stratify batch assignment; each
                class is split into ``num_batches`` roughly equal parts.
            num_batches: Number of batches to assign. Must be >= 1.

        Returns:
            SparkDataFrame: ``dataframe`` with an additional zero-based
            batch-id column. No Spark action is triggered; assignment
            stays lazy.

        Raises:
            ValueError: If ``num_batches`` < 1, if ``target_column`` is
                missing from ``dataframe``, or if the reserved internal
                batch-id column already exists.
        """
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
        """Yield pandas batches while materializing Spark assignment only once.

        The batch-assigned DataFrame is persisted once (memory and disk)
        before iteration and unpersisted when the generator is exhausted or
        closed, so the expensive window function is not recomputed for
        every batch filter.

        Args:
            dataframe: Input dataset to split into stratified batches.
            target_column: Column used to stratify batch assignment; see
                :meth:`assign_batches`.
            num_batches: Number of batches to assign and yield. Must be >= 1.

        Yields:
            pandas.DataFrame: One collected batch at a time, in batch-id
            order. Only the current batch is ever materialized in driver
            memory; empty batches are yielded as empty DataFrames and
            logged as a warning.

        Raises:
            ValueError: If ``num_batches`` < 1 or ``target_column`` is
                missing; see :meth:`assign_batches`.
        """
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
