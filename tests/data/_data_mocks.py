"""Small Spark datasets shared by backend integration tests."""

import random
from random import choice, randint, uniform
from typing import Any, Callable

from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import SparkSession
from pyspark.sql import Window as W
from pyspark.sql import functions as F
from pyspark.sql import types as T


def generate_obesity_row() -> dict[str, object]:
    """Generate one row for the multiclass obesity dataset."""
    return {
        "Age": float(randint(18, 60)),
        "Gender": choice(["Male", "Female"]),
        "Height": round(uniform(1.50, 1.90), 2),
        "Weight": round(uniform(45, 140), 1),
        "FAVC": choice(["yes", "no", "Frequently"]),
        "FCVC": round(uniform(1, 3), 1),
        "NCP": round(uniform(2, 5), 1),
        "SCC": choice(["yes", "no"]),
        "SMOKE": choice(["yes", "no"]),
        "CH2O": round(uniform(1, 3), 1),
        "family_history_with_overweight": choice(["yes", "no"]),
        "FAF": round(uniform(0, 3), 1),
        "TUE": round(uniform(0, 2), 1),
        "CAEC": choice(["no", "Sometimes", "Frequently", "Always"]),
        "CALC": round(uniform(0, 3), 1),
        "MTRANS": choice(["Walking", "Bike", "Automobile", "Public_Transportation"]),
        "NObeyesdad": choice(
            [
                "Underweight",
                "Normal_Weight",
                "Overweight_Level_I",
                "Overweight_Level_II",
                "Obesity_Type_I",
                "Obesity_Type_II",
                "Obesity_Type_III",
            ]
        ),
    }


def generate_diabetes_row() -> dict[str, object]:
    """Generate one row for the binary diabetes dataset."""
    return {
        "gender": choice(["Male", "Female"]),
        "age": float(randint(18, 90)),
        "hypertension": randint(0, 1),
        "heart_disease": randint(0, 1),
        "smoking_history": choice(["never", "current", "former", "ever", "No Info"]),
        "bmi": round(uniform(15, 45), 2),
        "HbA1c_level": round(uniform(3.5, 9.0), 1),
        "blood_glucose_level": randint(70, 250),
        "diabetes": randint(0, 1),
    }


OBESITY_SCHEMA = T.StructType(
    [
        T.StructField("Age", T.DoubleType(), False),
        T.StructField("Gender", T.StringType(), False),
        T.StructField("Height", T.DoubleType(), False),
        T.StructField("Weight", T.DoubleType(), False),
        T.StructField("FAVC", T.StringType(), False),
        T.StructField("FCVC", T.DoubleType(), False),
        T.StructField("NCP", T.DoubleType(), False),
        T.StructField("SCC", T.StringType(), False),
        T.StructField("SMOKE", T.StringType(), False),
        T.StructField("CH2O", T.DoubleType(), False),
        T.StructField("family_history_with_overweight", T.StringType(), False),
        T.StructField("FAF", T.DoubleType(), False),
        T.StructField("TUE", T.DoubleType(), False),
        T.StructField("CAEC", T.StringType(), False),
        T.StructField("CALC", T.DoubleType(), False),
        T.StructField("MTRANS", T.StringType(), False),
        T.StructField("NObeyesdad", T.StringType(), False),
    ]
)

DIABETES_SCHEMA = T.StructType(
    [
        T.StructField("gender", T.StringType(), False),
        T.StructField("age", T.DoubleType(), False),
        T.StructField("hypertension", T.IntegerType(), False),
        T.StructField("heart_disease", T.IntegerType(), False),
        T.StructField("smoking_history", T.StringType(), False),
        T.StructField("bmi", T.DoubleType(), False),
        T.StructField("HbA1c_level", T.DoubleType(), False),
        T.StructField("blood_glucose_level", T.IntegerType(), False),
        T.StructField("diabetes", T.IntegerType(), False),
    ]
)


def build_mock_spark_df(
    spark: SparkSession,
    size: int,
    row_generator: Callable[[], dict[str, object]],
    schema: T.StructType,
    seed: int | None = None,
) -> SparkDataFrame:
    """Build a deterministic Spark DataFrame from a row generator."""
    if seed is not None:
        random.seed(seed)
    return spark.createDataFrame(
        [row_generator() for _ in range(size)],
        schema=schema,
    )


def build_mock_obesity_df(
    spark: SparkSession, n: int, seed: int | None = None
) -> SparkDataFrame:
    return build_mock_spark_df(spark, n, generate_obesity_row, OBESITY_SCHEMA, seed)


def build_mock_diabetes_df(
    spark: SparkSession, n: int, seed: int | None = None
) -> SparkDataFrame:
    return build_mock_spark_df(spark, n, generate_diabetes_row, DIABETES_SCHEMA, seed)


def stratified_split_sparkdf(
    dataframe: SparkDataFrame,
    target_column: str,
    valid_size: float = 0.2,
    seed: int = 42,
) -> tuple[SparkDataFrame, SparkDataFrame]:
    """Create a stratified train/validation split."""
    if not 0 < valid_size < 1:
        raise ValueError("valid_size must be between 0 and 1")

    row_number = W.partitionBy(target_column).orderBy(F.rand(seed))
    numbered = dataframe.withColumn("_row_number", F.row_number().over(row_number))
    thresholds = dataframe.groupBy(target_column).agg(
        F.ceil(F.count(F.lit(1)) * valid_size).alias("_valid_count")
    )
    joined = numbered.join(thresholds, on=target_column, how="inner")
    temporary_columns = ("_row_number", "_valid_count")
    validation = joined.where(F.col("_row_number") <= F.col("_valid_count")).drop(
        *temporary_columns
    )
    train = joined.where(F.col("_row_number") > F.col("_valid_count")).drop(
        *temporary_columns
    )
    return train, validation


def assert_predicts_sample(
    model: Any,
    dataframe: SparkDataFrame,
    target_column: str,
    size: int = 5,
) -> None:
    """Assert that a backend model predicts every row in a small sample."""
    sample = dataframe.limit(size).toPandas()
    for column in sample.select_dtypes(include=["object"]):
        sample[column] = sample[column].astype("category")
    predictions = model.predict(sample.drop(columns=[target_column]))
    assert len(predictions) == len(sample)
