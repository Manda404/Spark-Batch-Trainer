"""Public package API with lazy backend imports."""

from importlib import import_module
from typing import TYPE_CHECKING, Any
from .factory import create_trainer

if TYPE_CHECKING:
    from .backends.catboost import CatBoostTrainer
    from .backends.lightgbm import LightGBMTrainer
    from .backends.xgboost import XGBoostTrainer

__all__ = [
    "XGBoostTrainer",
    "CatBoostTrainer",
    "LightGBMTrainer",
    "create_trainer",
    "__version__",
]

__version__ = "1.0.0"


_LAZY_EXPORTS = {
    "XGBoostTrainer": (".backends.xgboost", "XGBoostTrainer"),
    "CatBoostTrainer": (".backends.catboost", "CatBoostTrainer"),
    "LightGBMTrainer": (".backends.lightgbm", "LightGBMTrainer"),
}


def __getattr__(name: str) -> Any:
    """Import a model backend only when its trainer is requested."""
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
