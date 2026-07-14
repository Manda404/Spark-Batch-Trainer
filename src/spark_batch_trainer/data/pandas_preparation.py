"""Small, backend-neutral pandas preparation helpers."""

from logging import Logger
from typing import Dict, List, Tuple

from pandas import DataFrame as PandasDataFrame
from pandas.api.types import CategoricalDtype
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.types import StringType


class PandasDataPreparer:
    """Convert Spark data and normalize categorical pandas columns."""

    def __init__(self, logger: Logger) -> None:
        """Store the logger and start with an empty learned category schema.

        Args:
            logger: Logger used to report schema learning and conversion progress.
        """
        self._logger = logger
        self._category_schema: Dict[str, CategoricalDtype] = {}

    def reset_category_schema(self) -> None:
        """Clear categories learned for the previous training run."""
        self._category_schema.clear()

    def fit_category_schema(
        self,
        train_dataframe: SparkDataFrame,
        valid_dataframe: SparkDataFrame,
        target_column: str,
    ) -> List[str]:
        """Learn stable string categories across training and validation data.

        Every pandas batch must use identical category codes when a native
        backend continues training. The schema is therefore learned once from
        the union of train and validation values, then reused for every batch.

        Args:
            train_dataframe: Training dataset used to detect string feature columns.
            valid_dataframe: Validation dataset unioned with training values
                before computing distinct categories, so the schema covers both.
            target_column: Target column name, excluded from category
                learning even if it is string-typed.

        Returns:
            list[str]: Names of the string feature columns for which a
            stable category schema was learned.
        """
        string_columns = [
            field.name
            for field in train_dataframe.schema.fields
            if isinstance(field.dataType, StringType) and field.name != target_column
        ]
        self._category_schema = {}
        for column in string_columns:
            distinct_rows = (
                train_dataframe.select(column)
                .union(valid_dataframe.select(column))
                .distinct()
                .collect()
            )
            categories = sorted(
                (row[column] for row in distinct_rows if row[column] is not None),
                key=str,
            )
            self._category_schema[column] = CategoricalDtype(categories=categories)
        if string_columns:
            self._logger.info(
                "Learned stable categories for %d feature columns",
                len(string_columns),
            )
        return string_columns

    def collect(self, dataframe: SparkDataFrame, purpose: str) -> PandasDataFrame:
        """Collect a Spark DataFrame with an explicit driver-memory warning.

        Args:
            dataframe: Dataset to collect in full onto the driver.
            purpose: Short label describing the dataset (e.g.
                ``"validation"``), used only in the logged messages.

        Returns:
            pandas.DataFrame: The fully collected DataFrame.
        """
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
        """Convert object features to pandas categories and return their names.

        When a schema was learned via :meth:`fit_category_schema`, its
        columns and categories are reused so that batches share identical
        category codes. Otherwise, every ``object``-typed column other than
        ``target_column`` is converted using pandas' default categories.

        Args:
            dataframe: DataFrame whose object-typed columns should be
                converted. Converted in place.
            target_column: Target column name, excluded from conversion.
                Ignored when a category schema was already learned via
                :meth:`fit_category_schema`.

        Returns:
            tuple[pandas.DataFrame, list[str]]: The same DataFrame with
            categorical features converted, and the names of the columns
            that were converted to ``category`` dtype.
        """
        categorical_features = [
            name for name in self._category_schema if name in dataframe.columns
        ]
        if not categorical_features:
            categorical_features = [
                name
                for name in dataframe.select_dtypes(include=["object"]).columns
                if name != target_column
            ]
        if categorical_features:
            for name in categorical_features:
                category_type = self._category_schema.get(name, "category")
                dataframe[name] = dataframe[name].astype(category_type)
            self._logger.debug(
                "Converted categorical features: %s", categorical_features
            )
        return dataframe, categorical_features
