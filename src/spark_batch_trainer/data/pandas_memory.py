"""Pandas memory reduction used before native model training."""

from numpy import iinfo
from pandas import DataFrame


def downcast_numeric_features(dataframe: DataFrame) -> DataFrame:
    """Downcast wide numeric dtypes in place and return ``dataframe``.

    Categorical conversion happens earlier in the preparation pipeline, so
    this function only performs the two conversions that can still reduce a
    prepared batch: ``float64`` to ``float32`` and safe ``int64`` to ``int32``.
    """
    int32 = iinfo("int32")

    for name in dataframe.columns:
        dtype = dataframe[name].dtype
        if dtype == "float64":
            dataframe[name] = dataframe[name].astype("float32")
        elif dtype == "int64":
            minimum = dataframe[name].min()
            maximum = dataframe[name].max()
            if int32.min <= minimum and maximum <= int32.max:
                dataframe[name] = dataframe[name].astype("int32")

    return dataframe
