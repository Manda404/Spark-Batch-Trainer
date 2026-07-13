"""Public factory for selecting a model backend."""

from typing import Any


def create_trainer(backend: str) -> Any:
    """Create a trainer by a friendly backend name.

    Parameters
    ----------
    backend:
        One of ``"xgboost"``, ``"catboost"`` or ``"lightgbm"``. Common short
        aliases such as ``"xgb"`` and ``"lgbm"`` are accepted.
    """
    normalized = backend.lower().replace("-", "").replace("_", "")
    if normalized in {"xgb", "xgboost"}:
        from spark_batch_trainer.backends.xgboost import XGBoostTrainer

        return XGBoostTrainer()
    if normalized in {"cat", "catboost"}:
        from spark_batch_trainer.backends.catboost import CatBoostTrainer

        return CatBoostTrainer()
    if normalized in {"lgb", "lgbm", "lightgbm"}:
        from spark_batch_trainer.backends.lightgbm import LightGBMTrainer

        return LightGBMTrainer()
    raise ValueError(
        "unknown backend. Expected one of: 'xgboost', 'catboost', 'lightgbm'"
    )
