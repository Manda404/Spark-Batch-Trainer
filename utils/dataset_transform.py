"""Pandas preprocessing used by the example notebooks."""

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import LabelEncoder


@dataclass
class PreprocessedSplits:
    """Feature and target values prepared for model training."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_valid: pd.DataFrame | None = None
    y_valid: pd.Series | None = None
    X_test: pd.DataFrame | None = None
    y_test: pd.Series | None = None
    cat_cols: list[str] | None = None


def _split_features_target(
    dataframe: pd.DataFrame | None,
    target_column: str,
) -> tuple[pd.DataFrame | None, pd.Series | None]:
    if dataframe is None:
        return None, None
    return dataframe.drop(columns=[target_column]), dataframe[target_column]


def _align_categories(
    feature_sets: tuple[pd.DataFrame | None, ...],
    categorical_columns: list[str],
) -> None:
    for column in categorical_columns:
        categories = sorted(
            {
                str(value)
                for dataframe in feature_sets
                if dataframe is not None
                for value in dataframe[column].dropna().unique()
            }
        )
        dtype = CategoricalDtype(categories=categories)
        for dataframe in feature_sets:
            if dataframe is not None:
                dataframe[column] = dataframe[column].astype(str).astype(dtype)


def preprocess_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    target_column: str = "target",
    *,
    y_dtype: str | None = "int",
    cat_cols: Iterable[str] | None = None,
    copy: bool = True,
    auto_encode_target: bool = False,
) -> tuple[PreprocessedSplits, LabelEncoder | None]:
    """Prepare pandas train, validation, and test splits for the notebooks."""
    datasets = (train_df, valid_df, test_df)
    for name, dataframe in zip(("train", "valid", "test"), datasets):
        if dataframe is not None and target_column not in dataframe.columns:
            raise KeyError(f"[{name}] Missing target column '{target_column}'.")

    train, valid, test = (
        dataframe.copy() if dataframe is not None and copy else dataframe
        for dataframe in datasets
    )

    encoder = None
    if auto_encode_target:
        encoder = LabelEncoder().fit(train[target_column])
        for dataframe in (train, valid, test):
            if dataframe is not None:
                dataframe[target_column] = encoder.transform(dataframe[target_column])

    X_train, y_train = _split_features_target(train, target_column)
    X_valid, y_valid = _split_features_target(valid, target_column)
    X_test, y_test = _split_features_target(test, target_column)
    assert X_train is not None and y_train is not None

    if y_dtype is not None:
        y_train = y_train.astype(y_dtype)
        y_valid = y_valid.astype(y_dtype) if y_valid is not None else None
        y_test = y_test.astype(y_dtype) if y_test is not None else None

    categorical_columns = (
        list(cat_cols)
        if cat_cols is not None
        else X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    )
    _align_categories((X_train, X_valid, X_test), categorical_columns)

    return (
        PreprocessedSplits(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
            cat_cols=categorical_columns,
        ),
        encoder,
    )
