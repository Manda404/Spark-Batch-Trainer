import pandas as pd
from typing import Dict, Any
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F


class PandasLabelEncoder:
    """
    Encode/Decode a categorical target column into numeric labels with Pandas.
    """

    def __init__(self):
        self.class_to_index: Dict[Any, int] = {}
        self.index_to_class: Dict[int, Any] = {}
        self.fitted: bool = False

    def fit(self, df: pd.DataFrame, target_col: str) -> "PandasLabelEncoder":
        """Learn mapping from classes to integers."""
        classes = sorted(df[target_col].dropna().unique())
        self.class_to_index = {cls: idx for idx, cls in enumerate(classes)}
        self.index_to_class = {idx: cls for cls, idx in self.class_to_index.items()}
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Replace target column with encoded integers."""
        if not self.fitted:
            raise RuntimeError("PandasLabelEncoder must be fit() before transform().")
        df = df.copy()
        df[target_col] = df[target_col].map(self.class_to_index)
        return df

    def inverse_transform(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Replace encoded target column with original classes."""
        if not self.fitted:
            raise RuntimeError("PandasLabelEncoder must be fit() before inverse_transform().")
        df = df.copy()
        df[target_col] = df[target_col].map(self.index_to_class)
        return df

    def get_classes(self) -> Dict[int, Any]:
        """Return index-to-class mapping."""
        return self.index_to_class


class SparkLabelEncoder:
    """
    Encode/Decode a categorical target column into numeric labels with Spark.
    """

    def __init__(self):
        self.class_to_index: Dict[str, int] = {}
        self.index_to_class: Dict[int, str] = {}
        self.fitted: bool = False

    def fit(self, df: SparkDataFrame, target_col: str) -> "SparkLabelEncoder":
        """Learn mapping from classes to integers."""
        classes = df.select(target_col).distinct().rdd.flatMap(lambda x: x).collect()
        self.class_to_index = {cls: idx for idx, cls in enumerate(sorted(classes))}
        self.index_to_class = {idx: cls for cls, idx in self.class_to_index.items()}
        self.fitted = True
        return self

    def transform(self, df: SparkDataFrame, target_col: str) -> SparkDataFrame:
        """Replace target column with encoded integers."""
        if not self.fitted:
            raise RuntimeError("SparkLabelEncoder must be fit() before transform().")
        mapping_expr = F.create_map(
            [F.lit(x) for kv in self.class_to_index.items() for x in kv]
        )
        return df.withColumn(target_col, mapping_expr[F.col(target_col)])

    def inverse_transform(self, df: SparkDataFrame, target_col: str) -> SparkDataFrame:
        """Replace encoded target column with original classes."""
        if not self.fitted:
            raise RuntimeError("SparkLabelEncoder must be fit() before inverse_transform().")
        reverse_mapping_expr = F.create_map(
            [F.lit(x) for kv in self.index_to_class.items() for x in kv]
        )
        return df.withColumn(target_col, reverse_mapping_expr[F.col(target_col)])

    def get_classes(self) -> Dict[int, Any]:
        """Return index-to-class mapping."""
        return self.index_to_class