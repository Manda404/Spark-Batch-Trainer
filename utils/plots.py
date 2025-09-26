import matplotlib.pyplot as plt
from .metrics import compute_confusion


def plot_confusion(ax, y_true, y_pred, title, labels, class_labels, cmap):
    cm, cmn = compute_confusion(y_true, y_pred, labels)
    im = ax.imshow(cmn, interpolation="nearest", cmap=cmap)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
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


def plot_confusion_valid_and_test(
    results_valid_df, results_test_df, *, label_order, class_labels
):
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
