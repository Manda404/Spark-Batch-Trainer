import numpy as np
import pandas as pd


def make_results_df(y_true, y_pred) -> pd.DataFrame:
    """Return a standardized prediction-results DataFrame."""
    return pd.DataFrame(
        {"ground_true": np.asarray(y_true), "prediction": np.asarray(y_pred)}
    )
