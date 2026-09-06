"""General-based mathematical methods."""

from typing import List, Tuple, Union

import numpy as np

import opfython.math.distance as d
import opfython.utils.exception as e
from opfython.utils import logging

logger = logging.get_logger(__name__)


def _label_arrays(
    labels: Union[np.ndarray, List[int]], preds: Union[np.ndarray, List[int]]
) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    if labels.ndim != 1 or preds.ndim != 1 or labels.shape != preds.shape:
        raise e.SizeError(
            "`labels` and `preds` should be one-dimensional arrays "
            "with the same amount of samples"
        )
    return labels, preds


def confusion_matrix(
    labels: Union[np.array, List[int]], preds: Union[np.array, List[int]]
) -> np.array:
    """Calculates the confusion matrix between true and predicted labels.

    Args:
        labels: List or numpy array holding the true labels.
        preds: List or numpy array holding the predicted labels.

    Returns:
        (np.array): The confusion matrix.

    """

    labels, preds = _label_arrays(labels, preds)

    n_class = max(np.max(labels), np.max(preds)) + 1

    c_matrix = np.zeros((n_class, n_class))
    for label, pred in zip(labels, preds):
        c_matrix[label][pred] += 1

    return c_matrix


def normalize(array: np.array) -> np.array:
    """Standardizes an input array along its first axis.

    Args:
        array: Array to be normalized.

    Returns:
        (np.array): Z-scores with zero mean and unit standard deviation.

    """

    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)

    norm_array = (array - mean) / std

    return norm_array


def opf_accuracy(
    labels: Union[np.array, List[int]], preds: Union[np.array, List[int]]
) -> float:
    """Calculates the accuracy between true and predicted labels using OPF-style measure.

    Error rates with zero denominators contribute zero.

    Args:
        labels: List or numpy array holding the true labels.
        preds: List or numpy array holding the predicted labels.

    Returns:
        (float): The OPF accuracy measure between 0 and 1.

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


def opf_accuracy_per_label(
    labels: Union[np.array, List[int]], preds: Union[np.array, List[int]]
) -> np.ndarray:
    """Calculates the accuracy per label between true and predicted labels using OPF-style measure.

    Labels with no true samples have zero false-negative error and score 1.

    Args:
        labels: List or numpy array holding the true labels.
        preds: List or numpy array holding the predicted labels.

    Returns:
        (np.ndarray): OPF accuracies in label-index order, between 0 and 1.

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
    data: np.array, output: str, distance: str = "log_squared_euclidean"
) -> None:
    """Pre-computes a matrix of distances based on an input data.

    Args:
        data: Array of samples.
        output: File to be saved.
        distance: Distance metric to be used.

    """

    logger.info("Pre-computing distances ...")

    size = data.shape[0]

    distances = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            distances[i][j] = d.DISTANCES[distance](data[i], data[j])

    np.savetxt(output, distances)

    logger.info("Distances saved to: %s.", output)


def purity(
    labels: Union[np.array, List[int]], preds: Union[np.array, List[int]]
) -> float:
    """Calculate clustering purity independently of cluster numbering.

    Clusters may outnumber the true classes. Each cluster contributes the
    number of samples in its most frequent true class.

    Args:
        labels: List or numpy array holding the true labels.
        preds: List or numpy array holding the assigned labels by the clusters.

    Returns:
        (float): The purity measure.

    """

    labels, preds = _label_arrays(labels, preds)
    classes, label_indexes = np.unique(labels, return_inverse=True)
    clusters, cluster_indexes = np.unique(preds, return_inverse=True)
    counts = np.zeros((len(classes), len(clusters)), dtype=int)
    np.add.at(counts, (label_indexes, cluster_indexes), 1)

    return np.sum(np.max(counts, axis=0)) / len(labels)
