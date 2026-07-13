"""Small, backend-neutral pandas preparation helpers."""

from logging import Logger
from typing import List, Tuple

from pandas import DataFrame as PandasDataFrame
from pyspark.sql import DataFrame as SparkDataFrame


class PandasDataPreparer:
    """Convert Spark data and normalize categorical pandas columns."""

    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    def collect(self, dataframe: SparkDataFrame, purpose: str) -> PandasDataFrame:
        """Collect a Spark DataFrame with an explicit driver-memory warning."""
        self._logger.info("Collecting the complete %s DataFrame on the driver", purpose)
        result = dataframe.toPandas()
        self._logger.info(
            "Collected %s DataFrame with %d rows and %d columns",
            purpose,
            result.shape[0],
            result.shape[1],
        )
        return result

    def convert_categories(
        self,
        dataframe: PandasDataFrame,
        target_column: str = "",
    ) -> Tuple[PandasDataFrame, List[str]]:
        """Convert object features to pandas categories and return their names."""
        categorical_features = [
            name
            for name in dataframe.select_dtypes(include=["object"]).columns
            if name != target_column
        ]
        if categorical_features:
            dataframe[categorical_features] = dataframe[categorical_features].astype(
                "category"
            )
            self._logger.debug(
                "Converted categorical features: %s", categorical_features
            )
        return dataframe, categorical_features
