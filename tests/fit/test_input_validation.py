"""Spark-level input contract regression tests."""

import pytest
from pyspark.sql import types as T

from spark_batch_trainer import XGBoostTrainer


def test_rejects_more_batches_than_minority_rows(spark) -> None:
    dataframe = spark.createDataFrame(
        [(1.0, 0), (2.0, 0), (3.0, 1)],
        schema="feature double, target int",
    )
    trainer = XGBoostTrainer()

    with pytest.raises(ValueError, match="smallest training class"):
        trainer._validate_inputs(dataframe, dataframe, "target", num_batches=2)


def test_rejects_mismatched_feature_types(spark) -> None:
    train = spark.createDataFrame(
        [(1.0, 0), (2.0, 1)], schema="feature double, target int"
    )
    valid = spark.createDataFrame([(1, 0), (2, 1)], schema="feature int, target int")
    trainer = XGBoostTrainer()

    with pytest.raises(ValueError, match="column types differ"):
        trainer._validate_inputs(train, valid, "target", num_batches=1)


def test_rejects_null_target_values(spark) -> None:
    schema = T.StructType(
        [
            T.StructField("feature", T.DoubleType(), False),
            T.StructField("target", T.IntegerType(), True),
        ]
    )
    train = spark.createDataFrame([(1.0, 0), (2.0, 1)], schema=schema)
    valid = spark.createDataFrame([(1.0, 0), (2.0, None)], schema=schema)
    trainer = XGBoostTrainer()

    with pytest.raises(ValueError, match="must not contain nulls"):
        trainer._validate_inputs(train, valid, "target", num_batches=1)
