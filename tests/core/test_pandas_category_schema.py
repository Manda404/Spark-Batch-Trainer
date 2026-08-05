"""Tests for stable categorical encoding across collected pandas batches."""

import pandas as pd
from pandas.api.types import CategoricalDtype

from spark_batch_trainer.data.pandas_preparation import convert_categories


def test_convert_categories_reuses_fitted_category_order() -> None:
    schema = {"segment": CategoricalDtype(categories=["bronze", "gold", "silver"])}
    first = pd.DataFrame({"segment": ["gold", "bronze"], "target": [1, 0]})
    second = pd.DataFrame({"segment": ["silver", "gold"], "target": [0, 1]})

    convert_categories(first, schema)
    convert_categories(second, schema)

    assert first["segment"].cat.categories.tolist() == [
        "bronze",
        "gold",
        "silver",
    ]
    assert second["segment"].cat.categories.tolist() == [
        "bronze",
        "gold",
        "silver",
    ]
    assert first["segment"].cat.codes.tolist() == [1, 0]
    assert second["segment"].cat.codes.tolist() == [2, 1]
