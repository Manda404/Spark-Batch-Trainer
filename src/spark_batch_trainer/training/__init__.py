"""Backend-independent components of the batch-training workflow."""

from .config import TrainingConfig
from .history import TrainingHistory

__all__ = ["TrainingConfig", "TrainingHistory"]
