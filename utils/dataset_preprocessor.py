"""Dataset splitting for notebook workflows."""

from typing import Any

from pandas import DataFrame as PandasDataFrame
from sklearn.model_selection import train_test_split


def stratified_split_dataset(
    dataframe: PandasDataFrame,
    target_column: str,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    **kwargs: Any,
) -> tuple[PandasDataFrame, PandasDataFrame, PandasDataFrame]:
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
