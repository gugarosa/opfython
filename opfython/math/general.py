"""General-based mathematical methods.
"""

from typing import List, Optional, Union

import numpy as np

import opfython.math.distance as d
from opfython.utils import logging

logger = logging.get_logger(__name__)


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

    # Making sure that labels and predictions are arrays
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    # Calculating the number of classes
    n_class = np.max(labels) + 1

    # Creating an empty errors matrix
    c_matrix = np.zeros((n_class, n_class))

    for label, pred in zip(labels, preds):
        c_matrix[label][pred] += 1

    return c_matrix


def normalize(array: np.array) -> np.array:
    """Normalizes an input array.

    Args:
        array: Array to be normalized.

    Returns:
        (np.array): The normalized version (between 0 and 1) of the input array.

    """

    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)

    norm_array = (array - mean) / std

    return norm_array


def opf_accuracy(
    labels: Union[np.array, List[int]], preds: Union[np.array, List[int]]
) -> float:
    """Calculates the accuracy between true and predicted labels using OPF-style measure.

    Args:
        labels: List or numpy array holding the true labels.
        preds: List or numpy array holding the predicted labels.

    Returns:
        (float): The OPF accuracy measure between 0 and 1.

    """

    # Making sure that labels and predictions are arrays
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    # Calculating the number of classes
    n_class = np.max(labels) + 1

    # Creating an empty errors matrix
    errors = np.zeros((n_class, 2))

    # Gathering the amount of labels per class
    counts = np.bincount(labels)

    for label, pred in zip(labels, preds):
        if label != pred:
            errors[pred][0] += 1
            errors[label][1] += 1

    # Calculating the float value of the true label errors
    errors[:, 1] /= counts

    # Calculating the float value of the predicted label errors
    errors[:, 0] /= np.nansum(counts) - counts

    # Calculates the sum of errors per class
    errors = np.nansum(errors, axis=1)

    # Calculates the OPF accuracy
    accuracy = 1 - (np.sum(errors) / (2 * n_class))

    return accuracy


def opf_accuracy_per_label(
    labels: Union[np.array, List[int]], preds: Union[np.array, List[int]]
) -> float:
    """Calculates the accuracy per label between true and predicted labels using OPF-style measure.

    Args:
        labels: List or numpy array holding the true labels.
        preds: List or numpy array holding the predicted labels.

    Returns:
        (float): The OPF accuracy measure per label between 0 and 1.

    """

    # Making sure that labels and predictions are arrays
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    # Calculating the number of classes
    n_class = np.max(labels) + 1

    # Creating an empty errors array
    errors = np.zeros(n_class)

    # Gathering the amount of labels per class
    _, counts = np.unique(labels, return_counts=True)

    for label, pred in zip(labels, preds):
        if label != pred:
            errors[label] += 1

    # Calculating the float value of the true label errors
    errors /= counts

    # Calculates the OPF accuracy
    accuracy = 1 - errors

    return accuracy


def pre_compute_distance(
    data: np.array, output: str, distance: Optional[str] = "log_squared_euclidean"
) -> None:
    """Pre-computes a matrix of distances based on an input data.

    Args:
        data: Array of samples.
        output: File to be saved.
        distance: Distance metric to be used.

    """

    logger.info("Pre-computing distances ...")

    # Gathering the size of pre-computed matrix
    size = data.shape[0]

    # Creating an matrix of pre-computed distances
    distances = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            distances[i][j] = d.DISTANCES[distance](data[i], data[j])

    # Saves the distance matrix to an output
    np.savetxt(output, distances)

    logger.info("Distances saved to: %s.", output)


def purity(
    labels: Union[np.array, List[int]], preds: Union[np.array, List[int]]
) -> float:
    """Calculates the purity measure of an unsupervised technique.

    Args:
        labels: List or numpy array holding the true labels.
        preds: List or numpy array holding the assigned labels by the clusters.

    Returns:
        (float): The purity measure.

    """

    # Calculating the confusion matrix
    c_matrix = confusion_matrix(labels, preds)

    # Calculating the purity measure
    _purity = np.sum(np.max(c_matrix, axis=0)) / len(labels)

    return _purity
