"""Tests for reusable notebook dataset helpers."""

import pandas as pd
import pytest

from utils.dataset_loader import load_dataset
from utils.dataset_preprocessor import stratified_split_dataset


def test_load_dataset_accepts_project_relative_path() -> None:
    dataframe = load_dataset("data/multiclass_dataset/ObesityDataset.csv")

    assert not dataframe.empty
    assert "NObeyesdad" in dataframe.columns


def test_stratified_split_preserves_rows_and_classes() -> None:
    dataframe = pd.DataFrame(
        {
            "feature": range(100),
            "target": [0, 1] * 50,
        }
    )

    train, validation, test = stratified_split_dataset(
        dataframe,
        target_column="target",
        train_ratio=0.7,
        valid_ratio=0.15,
        test_ratio=0.15,
        random_state=42,
    )

    assert len(train) + len(validation) + len(test) == len(dataframe)
    assert all(split["target"].nunique() == 2 for split in (train, validation, test))


def test_stratified_split_rejects_invalid_ratios() -> None:
    dataframe = pd.DataFrame({"feature": range(10), "target": [0, 1] * 5})

    with pytest.raises(ValueError, match="must sum to 1.0"):
        stratified_split_dataset(
            dataframe,
            target_column="target",
            train_ratio=0.8,
            valid_ratio=0.15,
            test_ratio=0.15,
        )
