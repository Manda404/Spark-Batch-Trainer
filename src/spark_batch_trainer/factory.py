"""Public factory for selecting a model backend."""

from typing import Any

from spark_batch_trainer.training.base import BatchTrainer


def create_trainer(backend: str) -> BatchTrainer[Any]:
    """Create a trainer by a friendly backend name.

    Args:
        backend: One of ``"xgboost"``, ``"catboost"`` or ``"lightgbm"``.
            Common short aliases such as ``"xgb"`` and ``"lgbm"`` are
            accepted. Matching is case-insensitive and ignores ``-``/``_``
            separators.

    Returns:
        BatchTrainer: A new, untrained instance of the matching backend's
        trainer (:class:`~spark_batch_trainer.backends.xgboost.XGBoostTrainer`,
        :class:`~spark_batch_trainer.backends.catboost.CatBoostTrainer`, or
        :class:`~spark_batch_trainer.backends.lightgbm.LightGBMTrainer`).
        Only the requested backend's model SDK is imported.

    Raises:
        ValueError: If ``backend`` does not match a known backend name or alias.
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
