import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def make_results_df(y_true, y_pred) -> pd.DataFrame:
    """Return a standardized prediction-results DataFrame."""
    return pd.DataFrame(
        {"ground_true": np.asarray(y_true), "prediction": np.asarray(y_pred)}
    )


def compute_confusion(y_true, y_pred, labels):
    """Return raw and row-normalized confusion matrices."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cmn = cm / row_sums * 100.0
    return cm, cmn
