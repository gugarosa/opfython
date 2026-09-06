# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""General-based mathematical methods."""

from os import PathLike
from typing import TextIO

import numpy as np
from numpy.typing import ArrayLike

import opfython.math.distance as d
import opfython.utils.exception as e
from opfython.utils.logging import get_logger

logger = get_logger(__name__)


def _label_arrays(labels: ArrayLike, preds: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    if labels.ndim != 1 or preds.ndim != 1 or labels.shape != preds.shape:
        raise e.SizeError("`labels` and `preds` should be one-dimensional arrays with the same amount of samples.")

    return labels, preds


def confusion_matrix(labels: ArrayLike, preds: ArrayLike) -> np.ndarray:
    """Calculates the confusion matrix between true and predicted labels.

    Args:
        labels: Nonempty one-dimensional nonnegative integer true labels of shape (n_samples,).
        preds: Nonnegative integer predicted labels with the same shape as labels.

    Returns:
        np.ndarray: Float counts indexed by true-label rows and predicted-label columns, including absent label indexes.

    Raises:
        opfython.utils.exception.SizeError: Label and prediction arrays differ in shape or are not one-dimensional.

    Notes:
        Both axes span zero through the largest true or predicted label.

    """

    labels, preds = _label_arrays(labels, preds)

    n_class = max(np.max(labels), np.max(preds)) + 1

    c_matrix = np.zeros((n_class, n_class))
    for label, pred in zip(labels, preds):
        c_matrix[label][pred] += 1

    return c_matrix


def normalize(array: ArrayLike) -> np.ndarray:
    """Standardizes an input array along its first axis.

    Args:
        array: Numeric array-like input with at least one axis, standardized along axis zero.

    Returns:
        np.ndarray: Z-scores with the input shape and NumPy's arithmetic dtype, without changing the input.

    Raises:
        numpy.exceptions.AxisError: The input is scalar and has no axis zero.

    Notes:
        Zero standard deviations retain NumPy's division behavior rather than receiving a special-case replacement.

    """

    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)

    norm_array = (array - mean) / std

    return norm_array


def opf_accuracy(labels: ArrayLike, preds: ArrayLike) -> np.floating:
    """Calculates the accuracy between true and predicted labels using OPF-style measure.

    Args:
        labels: Nonempty one-dimensional nonnegative integer true labels of shape (n_samples,).
        preds: Nonnegative integer predicted labels with the same shape as labels.

    Returns:
        np.floating: OPF accuracy in [0, 1], averaged over indexes through the largest true or predicted label.

    Raises:
        opfython.utils.exception.SizeError: Label and prediction arrays differ in shape or are not one-dimensional.

    Notes:
        Error rates with zero denominators contribute zero.

    """

    labels, preds = _label_arrays(labels, preds)

    n_class = max(np.max(labels), np.max(preds)) + 1

    errors = np.zeros((n_class, 2))
    counts = np.bincount(labels, minlength=n_class)

    for label, pred in zip(labels, preds):
        if label != pred:
            errors[pred][0] += 1
            errors[label][1] += 1

    negatives = counts.sum() - counts
    np.divide(errors[:, 1], counts, out=errors[:, 1], where=counts != 0)
    np.divide(errors[:, 0], negatives, out=errors[:, 0], where=negatives != 0)
    errors = errors.sum(axis=1)

    accuracy = 1 - (np.sum(errors) / (2 * n_class))

    return accuracy


def opf_accuracy_per_label(labels: ArrayLike, preds: ArrayLike) -> np.ndarray:
    """Calculates the accuracy per label between true and predicted labels using OPF-style measure.

    Args:
        labels: Nonempty one-dimensional nonnegative integer true labels of shape (n_samples,).
        preds: Nonnegative integer predicted labels with the same shape as labels.

    Returns:
        np.ndarray: Float accuracies between zero and one, indexed from zero through the largest true label.

    Raises:
        opfython.utils.exception.SizeError: Label and prediction arrays differ in shape or are not one-dimensional.

    Notes:
        Labels with no true samples have zero false-negative error and score one.

    """

    labels, preds = _label_arrays(labels, preds)

    n_class = np.max(labels) + 1

    errors = np.zeros(n_class)
    counts = np.bincount(labels)

    for label, pred in zip(labels, preds):
        if label != pred:
            errors[label] += 1

    np.divide(errors, counts, out=errors, where=counts != 0)
    accuracy = 1 - errors

    return accuracy


def pre_compute_distance(
    data: np.ndarray, output: str | PathLike[str] | TextIO, distance: str = "log_squared_euclidean"
) -> None:
    """Saves all pairwise sample distances as a text matrix.

    Args:
        data: Feature array of shape (n_samples, n_features).
        output: Destination path or writable text stream for the (n_samples, n_samples) matrix.
        distance: Metric name registered in opfython.math.distance.DISTANCES.

    Raises:
        KeyError: The distance name is not registered when a sample pair is evaluated.
        OSError: The destination cannot be written.

    """

    logger.info("Pre-computing distances ...")

    size = data.shape[0]

    distances = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            distances[i][j] = d.DISTANCES[distance](data[i], data[j])

    np.savetxt(output, distances)

    logger.info("Distances saved to: %s.", output)


def purity(labels: ArrayLike, preds: ArrayLike) -> np.floating:
    """Calculate clustering purity independently of cluster numbering.

    Args:
        labels: Nonempty one-dimensional true class identifiers of shape (n_samples,).
        preds: Assigned cluster identifiers with the same shape as labels.

    Returns:
        np.floating: Fraction of samples belonging to the most frequent true class within their assigned cluster.

    Raises:
        opfython.utils.exception.SizeError: Label and prediction arrays differ in shape or are not one-dimensional.

    Notes:
        Clusters may outnumber the true classes and their identifiers need not match true class identifiers.

    """

    labels, preds = _label_arrays(labels, preds)

    classes, label_indexes = np.unique(labels, return_inverse=True)
    clusters, cluster_indexes = np.unique(preds, return_inverse=True)

    counts = np.zeros((len(classes), len(clusters)), dtype=int)
    np.add.at(counts, (label_indexes, cluster_indexes), 1)

    return np.sum(np.max(counts, axis=0)) / len(labels)
