import pandas as pd
from pandas import DataFrame as PandasDataFrame, Index, concat
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


# -------------------------------------------------------------------
# Categorical-feature helpers
# -------------------------------------------------------------------
def get_categorical_features(X: PandasDataFrame) -> Tuple[List[str], List[int]]:
    """
    Identify categorical features in a DataFrame.

    Args:
        X (pd.DataFrame): Input dataframe.

    Returns:
        Tuple:
            - cat_cols (List[str]): List of categorical column names.
            - cat_idx (List[int]): List of categorical column indices.
    """
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_idx = [X.columns.get_loc(col) for col in cat_cols]
    return cat_cols, cat_idx


def align_categoricals(
    X_train: PandasDataFrame,
    X_valid: Optional[PandasDataFrame],
    X_test: Optional[PandasDataFrame],
    cat_cols: List[str],
) -> Tuple[PandasDataFrame, Optional[PandasDataFrame], Optional[PandasDataFrame]]:
    """
    Align categorical feature levels across train/valid/test splits.

    Args:
        X_train (pd.DataFrame): Training dataset.
        X_valid (pd.DataFrame, optional): Validation dataset.
        X_test (pd.DataFrame, optional): Test dataset.
        cat_cols (List[str]): List of categorical columns.

    Returns:
        Tuple:
            - X_train (pd.DataFrame): Aligned training dataset.
            - X_valid (pd.DataFrame or None): Aligned validation dataset.
            - X_test (pd.DataFrame or None): Aligned test dataset.
    """
    for col in cat_cols:
        # Collect unique categories across available splits
        cats = Index(
            concat(
                [
                    df[col].astype(str)
                    for df in (X_train, X_valid, X_test)
                    if df is not None
                ],
                ignore_index=True,
            ).unique()
        ).sort_values()

        ctype = CategoricalDtype(categories=cats)

        for df in (X_train, X_valid, X_test):
            if df is not None:
                df[col] = df[col].astype(ctype)

    return X_train, X_valid, X_test


# -------------------------------------------------------------------
# Output data structure
# -------------------------------------------------------------------
@dataclass
class PreprocessedSplits:
    """Container for preprocessed dataset splits."""
    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: Optional[pd.DataFrame] = None
    y_valid: Optional[pd.Series] = None
    X_test: Optional[pd.DataFrame] = None
    y_test: Optional[pd.Series] = None
    cat_cols: Optional[List[str]] = None


# -------------------------------------------------------------------
# Public preprocessing workflow
# -------------------------------------------------------------------
def preprocess_splits(
    train_df: pd.DataFrame,
    valid_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    target_column: str = "target",
    *,
    y_dtype: Optional[str] = "int",   # "int", "float", "category", None
    cat_cols: Optional[Iterable[str]] = None,  # auto-detected if None
    cast_categoricals: bool = True,
    copy: bool = True,
    auto_encode_target: bool = False,
) -> Tuple[PreprocessedSplits, Optional[LabelEncoder]]:
    """
    Preprocess train/valid/test splits for ML training.

    Steps:
    - Optionally encode target labels with LabelEncoder.
    - Splits features (X) and target (y).
    - Casts target (y) to the desired dtype.
    - Detects or uses provided categorical columns.
    - Optionally casts categorical columns to dtype 'category'.
    - Aligns categorical levels across splits.

    Args:
        train_df (pd.DataFrame): Training dataset.
        valid_df (pd.DataFrame, optional): Validation dataset.
        test_df (pd.DataFrame, optional): Test dataset.
        target_column (str): Name of the target column. Defaults to "target".
        y_dtype (str, optional): Desired dtype for target labels. Defaults to "int".
        cat_cols (Iterable[str], optional): List of categorical columns. If None, detected automatically.
        cast_categoricals (bool): Whether to cast categorical columns to 'category'. Defaults to True.
        copy (bool): Whether to copy the dataframes to avoid modifying originals. Defaults to True.
        auto_encode_target (bool): If True, encodes the target column with LabelEncoder. Defaults to False.

    Returns:
        Tuple:
            - PreprocessedSplits: Dataclass containing X/y splits and categorical columns.
            - LabelEncoder or None: Fitted encoder if auto_encode_target=True, else None.
    """

    def ensure_target_exists(df: Optional[pd.DataFrame], name: str):
        if df is not None and target_column not in df.columns:
            raise KeyError(f"[{name}] Missing target column '{target_column}'.")

    # Validate target presence
    ensure_target_exists(train_df, "train")
    ensure_target_exists(valid_df, "valid")
    ensure_target_exists(test_df, "test")

    # Copy dataframes if required
    tr = train_df.copy() if copy else train_df
    va = valid_df.copy() if (valid_df is not None and copy) else valid_df
    te = test_df.copy() if (test_df is not None and copy) else test_df

    # Auto encode target labels if requested
    encoder = None
    if auto_encode_target:
        encoder = LabelEncoder()
        encoder.fit(tr[target_column])
        for df in (tr, va, te):
            if df is not None:
                df[target_column] = encoder.transform(df[target_column])

    # Split X / y
    def split_xy(df: Optional[pd.DataFrame]):
        if df is None:
            return None, None
        return df.drop(columns=[target_column]), df[target_column]

    X_train, y_train = split_xy(tr)
    X_valid, y_valid = split_xy(va)
    X_test, y_test = split_xy(te)

    # Cast y to desired dtype
    def cast_y(y: Optional[pd.Series]) -> Optional[pd.Series]:
        if y is None or y_dtype is None:
            return y
        return y.astype(y_dtype)

    y_train, y_valid, y_test = map(cast_y, [y_train, y_valid, y_test])

    # Detect categorical features if not provided
    if cat_cols is None:
        detected, _ = get_categorical_features(X_train)
        cat_cols = list(detected) if detected else []

    # Cast categorical columns to "category"
    if cast_categoricals and cat_cols:
        for df in (X_train, X_valid, X_test):
            if df is not None:
                for col in cat_cols:
                    if col in df.columns:
                        df[col] = df[col].astype("category")

    # Align categories across splits
    if cat_cols:
        X_train, X_valid, X_test = align_categoricals(X_train, X_valid, X_test, cat_cols)

    return PreprocessedSplits(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        X_test=X_test,
        y_test=y_test,
        cat_cols=list(cat_cols) if cat_cols else [],
    ), encoder
