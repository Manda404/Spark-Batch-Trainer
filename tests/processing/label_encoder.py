"""Target encoding helper for Spark integration tests."""

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F


def encode_target(
    train_dataframe: SparkDataFrame,
    valid_dataframe: SparkDataFrame,
    target_column: str,
) -> tuple[SparkDataFrame, SparkDataFrame, int]:
    """Encode target labels learned from the training DataFrame."""
    classes = sorted(
        train_dataframe.select(target_column)
        .distinct()
        .rdd.flatMap(lambda row: row)
        .collect()
    )
    mapping = F.create_map(
        [F.lit(value) for item in enumerate(classes) for value in reversed(item)]
    )

    def transform(dataframe: SparkDataFrame) -> SparkDataFrame:
        return dataframe.withColumn(target_column, mapping[F.col(target_column)])

    return transform(train_dataframe), transform(valid_dataframe), len(classes)
