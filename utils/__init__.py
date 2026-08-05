"""Reusable data and evaluation support for the project notebooks."""

from .dataset_loader import get_dataset_path, load_dataset
from .dataset_preprocessor import stratified_split_dataset
from .dataset_transform import preprocess_splits
from .metrics import make_results_df
from .visualize import (
    plot_binary_confusion_matrices,
    plot_class_distribution,
    plot_multiclass_confusion_matrices,
)

__all__ = [
    "get_dataset_path",
    "load_dataset",
    "stratified_split_dataset",
    "make_results_df",
    "plot_binary_confusion_matrices",
    "preprocess_splits",
    "plot_class_distribution",
    "plot_multiclass_confusion_matrices",
]
