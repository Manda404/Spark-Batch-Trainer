"""Tests for stable categorical encoding across collected pandas batches."""

import logging

import pandas as pd
from pandas.api.types import CategoricalDtype

from spark_batch_trainer.data.pandas_preparation import PandasDataPreparer


def test_convert_categories_reuses_fitted_category_order() -> None:
    preparer = PandasDataPreparer(logging.getLogger(__name__))
    preparer._category_schema = {
        "segment": CategoricalDtype(categories=["bronze", "gold", "silver"])
    }
    first = pd.DataFrame({"segment": ["gold", "bronze"], "target": [1, 0]})
    second = pd.DataFrame({"segment": ["silver", "gold"], "target": [0, 1]})

    prepared_first, columns_first = preparer.convert_categories(first, "target")
    prepared_second, columns_second = preparer.convert_categories(second, "target")

    assert columns_first == columns_second == ["segment"]
    assert prepared_first["segment"].cat.categories.tolist() == [
        "bronze",
        "gold",
        "silver",
    ]
    assert prepared_second["segment"].cat.categories.tolist() == [
        "bronze",
        "gold",
        "silver",
    ]
    assert prepared_first["segment"].cat.codes.tolist() == [1, 0]
    assert prepared_second["segment"].cat.codes.tolist() == [2, 1]
