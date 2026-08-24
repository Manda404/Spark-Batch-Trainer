import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def make_results_df(y_true: ArrayLike, y_pred: ArrayLike) -> pd.DataFrame:
    """Return a standardized prediction-results DataFrame."""
    return pd.DataFrame(
        {"ground_true": np.asarray(y_true), "prediction": np.asarray(y_pred)}
    )
