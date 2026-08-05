"""Spark-to-pandas collection and categorical preparation."""

from pandas import DataFrame
from pandas.api.types import CategoricalDtype
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import col, collect_set
from pyspark.sql.types import StringType

CategorySchema = dict[str, CategoricalDtype]


def learn_category_schema(
    train_dataframe: SparkDataFrame,
    valid_dataframe: SparkDataFrame,
    target_column: str,
) -> CategorySchema:
    """Learn stable string categories across train and validation in one action."""
    string_columns = [
        field.name
        for field in train_dataframe.schema.fields
        if isinstance(field.dataType, StringType) and field.name != target_column
    ]
    if not string_columns:
        return {}

    combined = train_dataframe.select(*string_columns).unionByName(
        valid_dataframe.select(*string_columns)
    )
    row = combined.agg(
        *(collect_set(col(name)).alias(name) for name in string_columns)
    ).first()
    schema = {
        name: CategoricalDtype(
            categories=sorted(
                (
                    ()
                    if row is None
                    else (value for value in row[name] if value is not None)
                ),
                key=str,
            )
        )
        for name in string_columns
    }
    return schema


def convert_categories(
    dataframe: DataFrame,
    schema: CategorySchema,
) -> None:
    """Apply the learned categorical dtypes to a pandas DataFrame in place."""
    for name, dtype in schema.items():
        if name in dataframe.columns:
            dataframe[name] = dataframe[name].astype(dtype)
