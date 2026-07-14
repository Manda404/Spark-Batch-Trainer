"""Reusable model-diagnostic visualizations for the example notebooks."""

from typing import Sequence, Tuple, Union
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from .metrics import compute_confusion

# Configure global visualization style
sns.set_theme(style="whitegrid")


# ==============================================================================
# 1. DATA DISTRIBUTION VISUALIZATIONS
# ==============================================================================

def plot_class_distribution(
    dataset: pd.DataFrame, 
    target_column: str, 
    figsize: Tuple[int, int] = (10, 5)
) -> pd.DataFrame:
    """
    Plots a bar chart of the target class distribution with distinct colors 
    and dynamic labels, and returns the calculated distribution DataFrame.

    Parameters
    ----------
    dataset : pandas.DataFrame
        The input dataset containing the target column.
    target_column : str
        The name of the target column to analyze.
    figsize : tuple of int, default (10, 5)
        Width and height of the resulting figure.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the row counts and percentage share per class.
    """
    # 1. Distribution Calculations
    class_distribution = (
        dataset[target_column]
        .value_counts()
        .rename_axis("class_name")
        .reset_index(name="rows")
    )
    class_distribution["share"] = class_distribution["rows"] / len(dataset)

    # 2. Visualization Creation
    plt.figure(figsize=figsize)
    
    ax = sns.barplot(
        data=class_distribution, 
        x="class_name", 
        y="rows", 
        hue="class_name",
        palette="viridis",
        legend=False
    )
    
    plt.title("Target-Class Distribution", fontweight="bold", fontsize=14, pad=15)
    plt.xlabel("Class", fontsize=11, labelpad=10)
    plt.ylabel("Rows", fontsize=11, labelpad=10)
    plt.xticks(rotation=35, ha="right")
    
    max_height = class_distribution["rows"].max()
    plt.ylim(0, max_height * 1.15)

    # 3. Adding Text Labels (Count & Percentage)
    for i, row in class_distribution.iterrows():
        count = int(row["rows"])
        share = row["share"] * 100
        label = f"{count:,}\n({share:.2f}%)"
        
        ax.text(
            x=i, 
            y=count + (max_height * 0.01),
            s=label, 
            ha="center", 
            va="bottom", 
            fontsize=10, 
            fontweight="bold"
        )

    plt.tight_layout()
    plt.show()

    return class_distribution


# ==============================================================================
# 2. CORE CONFUSION MATRIX PLOTTER
# ==============================================================================

def plot_confusion(
    ax: plt.Axes, 
    y_true: Union[Sequence, pd.Series], 
    y_pred: Union[Sequence, pd.Series], 
    title: str, 
    labels: Sequence, 
    class_labels: Sequence[str], 
    cmap: str
) -> None:
    """
    Plots a single normalized confusion matrix with raw counts and percentages 
    annotated inside each cell.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The specific subplot axis where the matrix heatmap will be drawn.
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated targets as returned by a classifier.
    title : str
        The title text to display on top of the subplot.
    labels : list or array-like
        List of labels to index the matrix. This controls the ordering of the classes.
    class_labels : list of str
        Human-readable class names to display on the x and y axes ticks.
    cmap : str
        The mapping from data values to color space (e.g., 'Blues', 'Oranges').
    """
    cm, cmn = compute_confusion(y_true, y_pred, labels)
    im = ax.imshow(cmn, interpolation="nearest", cmap=cmap)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted label", fontsize=11, labelpad=8)
    ax.set_ylabel("True label", fontsize=11, labelpad=8)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(class_labels)
    ax.set_yticklabels(class_labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cmn[i, j]:.1f}%)",
                ha="center",
                va="center",
                fontsize=11,
                color="black",
            )

    plt.colorbar(im, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)


# ==============================================================================
# 3. HIGH-LEVEL PIPELINE VISUALIZATIONS
# ==============================================================================

def plot_binary_confusion_matrices(
    results_valid_df: pd.DataFrame, 
    results_test_df: pd.DataFrame, 
    *, 
    label_order: Sequence, 
    class_labels: Sequence[str]
) -> None:
    """
    Generates a side-by-side comparison of row-normalized confusion matrices 
    for both validation and test datasets (Binary context).

    Parameters
    ----------
    results_valid_df : pandas.DataFrame
        DataFrame containing validation results. Must include 'ground_true' 
        and 'prediction' columns.
    results_test_df : pandas.DataFrame
        DataFrame containing test results. Must include 'ground_true' 
        and 'prediction' columns.
    label_order : list or tuple
        The exact numerical order/sequence of class IDs (e.g., (0, 1)).
    class_labels : list or tuple of str
        Custom text strings representing the class labels on the axes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    plot_confusion(
        axes[0],
        results_valid_df["ground_true"],
        results_valid_df["prediction"],
        "Confusion Matrix — Validation (Row-Normalized)",
        label_order,
        class_labels,
        cmap="Blues",
    )
    plot_confusion(
        axes[1],
        results_test_df["ground_true"],
        results_test_df["prediction"],
        "Confusion Matrix — Test (Row-Normalized)",
        label_order,
        class_labels,
        cmap="Oranges",
    )
    plt.tight_layout()
    plt.show()


def plot_multiclass_confusion_matrices(
    y_true_valid: Union[Sequence, pd.Series],
    y_pred_valid: Union[Sequence, pd.Series],
    y_true_test: Union[Sequence, pd.Series],
    y_pred_test: Union[Sequence, pd.Series],
    mapping_df: pd.DataFrame,
    normalize: bool = False,
) -> None:
    """
    Plots multiclass validation and test confusion matrices side-by-side.
    Uses sorted class names mapped from a reference mapping DataFrame.

    Parameters
    ----------
    y_true_valid : array-like
        Ground truth target values for the validation set.
    y_pred_valid : array-like
        Predicted target values for the validation set.
    y_true_test : array-like
        Ground truth target values for the test set.
    y_pred_test : array-like
        Predicted target values for the test set.
    mapping_df : pandas.DataFrame
        Reference DataFrame containing 'class_id' and 'class_name' columns 
        to ensure proper ordering and labeling.
    normalize : bool, default False
        Whether to compute row-normalized confusion matrices displayed in percentages.
    """
    # 1. Extract Ordered Class Names From Mapping
    sorted_mapping = mapping_df.sort_values("class_id")
    class_names = sorted_mapping["class_name"].tolist()
    labels = sorted_mapping["class_id"].tolist()

    # 2. Compute Confusion Matrices
    normalization = "true" if normalize else None
    matrices = (
        confusion_matrix(y_true_valid, y_pred_valid, labels=labels, normalize=normalization),
        confusion_matrix(y_true_test, y_pred_test, labels=labels, normalize=normalization),
    )
    
    number_format = ".2%" if normalize else "g"
    figure, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 3. Plot Heatmaps
    for axis, matrix, title, color in zip(
        axes,
        matrices,
        ("Validation Set", "Test Set"),
        ("Blues", "Greens"),
    ):
        sns.heatmap(
            matrix,
            annot=True,
            fmt=number_format,
            cmap=color,
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
            ax=axis,
            annot_kws={"size": 10, "weight": "bold"}
        )
        axis.set_title(title, fontweight="bold", fontsize=12, pad=10)
        axis.set_xlabel("Predicted Class", fontsize=10, labelpad=8)
        axis.set_ylabel("True Class", fontsize=10, labelpad=8)
        axis.tick_params(axis="x", rotation=45)
        axis.tick_params(axis="y", rotation=0)
        
    figure.suptitle("Multiclass Validation and Test Confusion Matrices", fontweight="bold", fontsize=14, y=1.02)
    figure.tight_layout()
    plt.show()