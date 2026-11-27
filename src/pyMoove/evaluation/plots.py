"""Plots for evaluating binary classifiers."""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    precision_recall_curve,
)


def _get_axes(ax: Axes | None) -> Axes:
    if ax is None:
        _, ax = plt.subplots()
    return ax


def plot_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    ax: Axes | None = None,
    labels: ArrayLike | None = None,
    display_labels: ArrayLike | None = None,
) -> Axes:
    """Plot a confusion matrix and return its Matplotlib axes.

    ``y_pred`` should contain predicted class labels, for example probabilities
    converted to labels using the chosen decision threshold.
    """
    ax = _get_axes(ax)
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        display_labels=display_labels,
        cmap="Blues",
        values_format="d",
        colorbar=False,
        ax=ax,
    )
    return ax


def plot_precision_recall_curve(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    positive_label: object = 1,
    ax: Axes | None = None,
) -> Axes:
    """Plot precision against recall over score thresholds."""
    ax = _get_axes(ax)
    PrecisionRecallDisplay.from_predictions(
        y_true,
        y_score,
        pos_label=positive_label,
        ax=ax,
    )
    return ax


def plot_recall_sensitivity_curve(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    positive_label: object = 1,
    ax: Axes | None = None,
) -> Axes:
    """Plot recall (sensitivity) against score thresholds."""
    ax = _get_axes(ax)
    recall, thresholds = precision_recall_curve(
        y_true,
        y_score,
        pos_label=positive_label,
    )[1:]
    ax.plot(thresholds, recall[:-1])
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Recall (sensitivity)")
    ax.set_title("Recall (sensitivity) by threshold")
    return ax


def plot_roc_curve(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    positive_label: object = 1,
    ax: Axes | None = None,
) -> Axes:
    """Plot the true-positive rate against the false-positive rate."""
    ax = _get_axes(ax)
    RocCurveDisplay.from_predictions(
        y_true,
        y_score,
        pos_label=positive_label,
        ax=ax,
    )
    return ax
