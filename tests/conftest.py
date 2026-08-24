import sys
from unittest.mock import patch

import pytest
from pyspark.sql import SparkSession

from tests.data._data_mocks import (
    build_mock_diabetes_df,
    build_mock_obesity_df,
    stratified_split_sparkdf,
)


@pytest.fixture(scope="session")
def spark():
    """Create one reusable Spark session for the complete test suite."""
    python_environment = {
        "PYSPARK_PYTHON": sys.executable,
        "PYSPARK_DRIVER_PYTHON": sys.executable,
    }
    with patch.dict("os.environ", python_environment):
        spark = (
            SparkSession.builder.master("local[*]")
            .appName("SparkBatchTrainerTests")
            .getOrCreate()
        )
        yield spark
        spark.stop()


@pytest.fixture
def mock_data_obesity(spark):
    """Build a multiclass train/validation dataset."""
    dataframe = build_mock_obesity_df(spark, n=200, seed=42)
    train, valid = stratified_split_sparkdf(
        dataframe, target_column="NObeyesdad", valid_size=0.2, seed=42
    )
    return train, valid, "NObeyesdad"


@pytest.fixture
def mock_data_diabetes(spark):
    """Build a binary train/validation dataset."""
    dataframe = build_mock_diabetes_df(spark, n=200, seed=42)
    train, valid = stratified_split_sparkdf(
        dataframe, target_column="diabetes", valid_size=0.2, seed=42
    )
    return train, valid, "diabetes"


@pytest.fixture
def config_training():
    """Provide the small configuration shared by integration tests."""
    return {"num_batches": 2, "show_learning_curve": False}
