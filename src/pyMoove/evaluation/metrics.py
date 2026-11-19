"""Metrics for evaluating binary classifiers."""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.multiclass import type_of_target


def evaluate_binary_classifier(
    y_true: ArrayLike,
    y_probability: ArrayLike,
    *,
    threshold: float = 0.5,
    positive_label: object = 1,
) -> dict[str, int | float]:
    """Calculate threshold-based and ranking metrics for binary predictions.

    ``y_probability`` must contain the predicted probability for the positive
    class, not the classifier's predicted labels or a two-column probability
    matrix.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_probability:
        Predicted probabilities for the positive class, in the range [0, 1].
    threshold:
        Probability threshold used to classify a sample as positive.
    positive_label:
        Value representing the positive class in ``y_true``.

    Returns
    -------
    dict[str, int | float]
        Sample and confusion-matrix counts, threshold-based metrics
        (precision, recall, specificity, F1, and balanced accuracy), ROC-AUC,
        and average precision.

    Raises
    ------
    ValueError
        If the labels or probabilities are empty, not one-dimensional, have
        different lengths, are not binary, or contain invalid probabilities.
    """
    true_labels = np.asarray(y_true)
    probabilities = np.asarray(y_probability, dtype=float)

    if true_labels.ndim != 1 or probabilities.ndim != 1:
        raise ValueError("y_true and y_probability must be one-dimensional")
    if len(true_labels) == 0:
        raise ValueError("y_true and y_probability must not be empty")
    if len(true_labels) != len(probabilities):
        raise ValueError("y_true and y_probability must have the same length")
    if type_of_target(true_labels) != "binary":
        raise ValueError("y_true must contain exactly two non-missing classes")
    if positive_label not in np.unique(true_labels):
        raise ValueError("positive_label must be present in y_true")
    if not np.isfinite(probabilities).all() or (
        (probabilities < 0) | (probabilities > 1)
    ).any():
        raise ValueError("y_probability values must be finite and in [0, 1]")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and in [0, 1]")

    binary_labels = (true_labels == positive_label).astype(np.uint8)
    predictions = (probabilities >= threshold).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(
        binary_labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "n_samples": int(len(true_labels)),
        "n_positive": int(np.count_nonzero(binary_labels)),
        "n_negative": int(len(binary_labels) - np.count_nonzero(binary_labels)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision_score(binary_labels, predictions, zero_division=0)),
        "recall": float(recall_score(binary_labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(binary_labels, predictions, zero_division=0)),
        "balanced_accuracy": float(
            balanced_accuracy_score(binary_labels, predictions)
        ),
        "roc_auc": float(roc_auc_score(binary_labels, probabilities)),
        "average_precision": float(
            average_precision_score(binary_labels, probabilities)
        ),
    }
