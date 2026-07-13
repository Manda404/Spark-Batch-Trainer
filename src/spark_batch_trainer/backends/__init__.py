"""Model-specific training backends.

Backends are imported lazily so users only need the optional dependency of the
trainer they instantiate.
"""

from importlib import import_module
from typing import Any

__all__ = ["XGBoostTrainer", "CatBoostTrainer", "LightGBMTrainer"]

_LAZY_EXPORTS = {
    "XGBoostTrainer": (".xgboost", "XGBoostTrainer"),
    "CatBoostTrainer": (".catboost", "CatBoostTrainer"),
    "LightGBMTrainer": (".lightgbm", "LightGBMTrainer"),
}


def __getattr__(name: str) -> Any:
    """Load only the requested model backend."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
