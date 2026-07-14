"""Reusable data and evaluation support for the project notebooks."""

from .dataset_loader import find_project_root, get_dataset_path, load_dataset, get_data_split
from .dataset_preprocessor import (
    make_train_valid_split,
    to_spark_dfs,
    stop_spark_session,
    stratified_split_dataset,
)
from .metrics import make_results_df
from .dataset_transform import preprocess_splits
from .visualize import (plot_class_distribution,plot_multiclass_confusion_matrices, plot_binary_confusion_matrices)


__all__ = [
    "find_project_root",
    "get_dataset_path",
    "load_dataset",
    "get_data_split",
    "make_train_valid_split",
    "to_spark_dfs",
    "stop_spark_session",
    "stratified_split_dataset",
    "make_results_df",
    "plot_binary_confusion_matrices",
    "preprocess_splits",
    "plot_class_distribution",
    "plot_multiclass_confusion_matrices"
]
