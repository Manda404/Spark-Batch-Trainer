"""Deterministic, target-stratified Spark batch collection."""

from logging import Logger
from typing import Generator, cast

from pandas import DataFrame as PandasDataFrame
from pyspark import StorageLevel
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import Window
from pyspark.sql.functions import col, ntile, rand

_BATCH_COLUMN = "__spark_batch_trainer_batch_id"


def iter_pandas_batches(
    dataframe: SparkDataFrame,
    target_column: str,
    num_batches: int,
    logger: Logger,
    *,
    seed: int = 42,
) -> Generator[PandasDataFrame, None, None]:
    """Assign stratified batches in Spark and collect them one at a time."""
    if _BATCH_COLUMN in dataframe.columns:
        raise ValueError(f"reserved internal column already exists: {_BATCH_COLUMN}")

    window = Window.partitionBy(target_column).orderBy(rand(seed=seed))
    assigned = dataframe.withColumn(
        _BATCH_COLUMN, ntile(num_batches).over(window) - 1
    ).persist(StorageLevel.MEMORY_AND_DISK)
    try:
        for batch_id in range(num_batches):
            logger.info("Collecting batch %d/%d", batch_id + 1, num_batches)
            batch = cast(
                PandasDataFrame,
                assigned.filter(col(_BATCH_COLUMN) == batch_id)
                .drop(_BATCH_COLUMN)
                .toPandas(),
            )
            if batch.empty:
                raise RuntimeError(
                    f"batch {batch_id + 1} is empty; reduce num_batches or "
                    "inspect target stratification"
                )
            yield batch
    finally:
        assigned.unpersist(blocking=False)
