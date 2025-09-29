import pytest
from pyspark.sql import SparkSession
from tests.spark_config import settings


@pytest.fixture(scope="session")
def spark():
    """Fixture Spark réutilisable pour tous les tests."""
    spark = (
        SparkSession.builder
        .master(settings.SPARK_MASTER)
        .appName(settings.SPARK_APP_NAME)
        .getOrCreate()
    )
    yield spark
    spark.stop()
