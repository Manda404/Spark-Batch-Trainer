import pytest
from pyspark.sql import SparkSession
from tests.spark_config import settings


@pytest.fixture(scope="session")
def spark():
    """Create one reusable Spark session for the complete test suite."""
    spark = (
        SparkSession.builder
        .master(settings.SPARK_MASTER)
        .appName(settings.SPARK_APP_NAME)
        .getOrCreate()
    )
    yield spark
    spark.stop()
