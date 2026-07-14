"""Tests for notebook visualization helpers."""

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt

from utils.plots import plot_model_confusion_comparison


def test_plot_model_confusion_comparison_returns_one_axis_per_model(
    monkeypatch,
) -> None:
    monkeypatch.setattr(plt, "show", lambda: None)
    results = [
        {"model": "Model A", "predictions": [0, 1, 1, 0]},
        {"model": "Model B", "predictions": [0, 0, 1, 1]},
    ]

    figure = plot_model_confusion_comparison(
        [0, 1, 1, 0],
        results,
        class_labels=["Negative", "Positive"],
    )

    assert len(figure.axes) == 2
    assert [axis.get_title() for axis in figure.axes] == ["Model A", "Model B"]
    plt.close(figure)
