import random
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql import types as T
from pyspark.sql import functions as F
from pyspark.sql import Window as W
from random import randint, choice, uniform

# =========================================================
# 1. Dataclasses
# =========================================================

@dataclass
class ObesityRow:
    Age: float
    Gender: str
    Height: float
    Weight: float
    FAVC: str
    FCVC: float
    NCP: float
    SCC: str
    SMOKE: str
    CH2O: float
    family_history_with_overweight: str
    FAF: float
    TUE: float
    CAEC: str
    CALC: float
    MTRANS: str
    NObeyesdad: str


@dataclass
class DiabetesRow:
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: float
    diabetes: int   # binaire


# =========================================================
# 2. Row Generators
# =========================================================

def generate_obesity_row() -> ObesityRow:
    """Generate one mock row for the multiclass obesity dataset."""
    return ObesityRow(
        Age=float(randint(18, 60)),
        Gender=choice(["Male", "Female"]),
        Height=round(uniform(1.50, 1.90), 2),
        Weight=round(uniform(45, 140), 1),
        FAVC=choice(["yes", "no", "Frequently"]),
        FCVC=round(uniform(1, 3), 1),
        NCP=round(uniform(2, 5), 1),
        SCC=choice(["yes", "no"]),
        SMOKE=choice(["yes", "no"]),
        CH2O=round(uniform(1, 3), 1),
        family_history_with_overweight=choice(["yes", "no"]),
        FAF=round(uniform(0, 3), 1),
        TUE=round(uniform(0, 2), 1),
        CAEC=choice(["no", "Sometimes", "Frequently", "Always"]),
        CALC=round(uniform(0, 3), 1),
        MTRANS=choice(["Walking", "Bike", "Automobile", "Public_Transportation"]),
        NObeyesdad=choice([
            "Underweight",
            "Normal_Weight",
            "Overweight_Level_I",
            "Overweight_Level_II",
            "Obesity_Type_I",
            "Obesity_Type_II",
            "Obesity_Type_III",
        ]),
    )


def generate_diabetes_row() -> DiabetesRow:
    """Generate one mock row for the binary diabetes dataset."""
    return DiabetesRow(
        gender=choice(["Male", "Female"]),
        age=float(randint(18, 90)),
        hypertension=randint(0, 1),
        heart_disease=randint(0, 1),
        smoking_history=choice(["never", "current", "former", "ever", "No Info"]),
        bmi=round(uniform(15, 45), 2),
        HbA1c_level=round(uniform(3.5, 9.0), 1),
        blood_glucose_level=randint(70, 250),
        diabetes=randint(0, 1),
    )


# =========================================================
# 3. Builders for Spark DataFrames
# =========================================================

def build_mock_spark_df(
    spark: SparkSession,
    n: int,
    row_generator,
    schema: T.StructType,
    seed: Optional[int] = None
) -> SparkDataFrame:
    """
    Build a Spark DataFrame from a row generator.
    """
    if seed is not None:
        random.seed(seed)
    rows = [asdict(row_generator()) for _ in range(n)]
    return spark.createDataFrame(rows, schema=schema)


# Explicit Spark schemas.
OBESITY_SCHEMA = T.StructType([
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
])

DIABETES_SCHEMA = T.StructType([
    T.StructField("gender", T.StringType(), False),
    T.StructField("age", T.DoubleType(), False),
    T.StructField("hypertension", T.IntegerType(), False),
    T.StructField("heart_disease", T.IntegerType(), False),
    T.StructField("smoking_history", T.StringType(), False),
    T.StructField("bmi", T.DoubleType(), False),
    T.StructField("HbA1c_level", T.DoubleType(), False),
    T.StructField("blood_glucose_level", T.IntegerType(), False),
    T.StructField("diabetes", T.IntegerType(), False),
])


# Dataset-specific helpers.
def build_mock_obesity_df(spark: SparkSession, n: int, seed: Optional[int] = None) -> SparkDataFrame:
    return build_mock_spark_df(spark, n, generate_obesity_row, OBESITY_SCHEMA, seed)


def build_mock_diabetes_df(spark: SparkSession, n: int, seed: Optional[int] = None) -> SparkDataFrame:
    return build_mock_spark_df(spark, n, generate_diabetes_row, DIABETES_SCHEMA, seed)


# =========================================================
# 4. Stratified Split
# =========================================================

def stratified_split_sparkdf(
    sparkdf: SparkDataFrame,
    target_col: str,
    valid_size: float = 0.2,
    seed: int = 42
) -> Tuple[SparkDataFrame, SparkDataFrame]:
    """
    Create a stratified train/validation split while preserving class ratios.
    """
    if not (0.0 < valid_size < 1.0):
        raise ValueError("valid_size must be between 0 and 1")

    w = W.partitionBy(target_col).orderBy(F.rand(seed))
    df_rn = sparkdf.withColumn("_rn", F.row_number().over(w))

    class_counts = sparkdf.groupBy(target_col).agg(F.count(F.lit(1)).alias("_cnt"))
    thresholds = class_counts.withColumn("_valid_k", F.ceil(F.col("_cnt") * valid_size))

    df_join = df_rn.join(thresholds, on=target_col, how="inner")

    valid_df = df_join.where(F.col("_rn") <= F.col("_valid_k")).drop("_rn", "_cnt", "_valid_k")
    train_df = df_join.where(F.col("_rn") > F.col("_valid_k")).drop("_rn", "_cnt", "_valid_k")

    return train_df, valid_df
