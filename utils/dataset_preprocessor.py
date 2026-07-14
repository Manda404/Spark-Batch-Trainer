"""Splitting and Spark-conversion helpers for notebook workflows."""

from typing import Any, Optional, Tuple

from pandas import DataFrame as PandasDataFrame
from pandas import concat
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split


def make_train_valid_split(
    dataframe: PandasDataFrame,
    target: str,
    train_size: float = 0.8,
    val_size: float = 0.2,
    **kwargs: Any,
) -> Tuple[PandasDataFrame, PandasDataFrame]:
    """Create a stratified train/validation split with original column order."""
    if target not in dataframe.columns:
        raise KeyError(f"Target column not found: {target}")
    if abs((train_size + val_size) - 1.0) > 1e-9:
        raise ValueError("train_size and val_size must sum to 1.0")

    features = dataframe.drop(columns=[target])
    labels = dataframe[target]
    kwargs.setdefault("stratify", labels)
    train_features, valid_features, train_labels, valid_labels = train_test_split(
        features,
        labels,
        train_size=train_size,
        test_size=val_size,
        **kwargs,
    )
    train = concat([train_features, train_labels], axis=1)[dataframe.columns]
    validation = concat([valid_features, valid_labels], axis=1)[dataframe.columns]
    return train, validation


def stratified_split_dataset(
    dataframe: PandasDataFrame,
    target_column: str,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    **kwargs: Any,
) -> Tuple[PandasDataFrame, PandasDataFrame, PandasDataFrame]:
    """Create reproducible train, validation, and test stratified splits."""
    if target_column not in dataframe.columns:
        raise KeyError(f"Target column not found: {target_column}")
    if min(train_ratio, valid_ratio, test_ratio) <= 0:
        raise ValueError("all split ratios must be greater than zero")
    if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-9:
        raise ValueError("train_ratio, valid_ratio, and test_ratio must sum to 1.0")

    train, temporary = train_test_split(
        dataframe,
        train_size=train_ratio,
        stratify=dataframe[target_column],
        random_state=random_state,
        **kwargs,
    )
    relative_test_ratio = test_ratio / (valid_ratio + test_ratio)
    validation, test = train_test_split(
        temporary,
        test_size=relative_test_ratio,
        stratify=temporary[target_column],
        random_state=random_state,
        **kwargs,
    )
    return train.copy(), validation.copy(), test.copy()


def to_spark_dfs(
    train_df: PandasDataFrame,
    valid_df: PandasDataFrame,
    spark: Optional[SparkSession] = None,
    app_name: str = "SparkBatchTrainerNotebook",
    **kwargs: Any,
) -> Tuple[SparkDataFrame, SparkDataFrame]:
    """Convert pandas training and validation splits to Spark DataFrames."""
    active_spark = spark or SparkSession.builder.appName(app_name).getOrCreate()
    return (
        active_spark.createDataFrame(train_df, **kwargs),
        active_spark.createDataFrame(valid_df, **kwargs),
    )


def stop_spark_session(spark: Optional[SparkSession]) -> None:
    """Stop a Spark session when one is active."""
    if spark is not None:
        spark.stop()
