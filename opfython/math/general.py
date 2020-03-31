import numpy as np

import opfython.math.distance as d
import opfython.utils.logging as l

logger = l.get_logger(__name__)


def confusion_matrix(labels, preds):
    """Calculates the confusion matrix between true and predicted labels.

    Args:
        labels (np.array | list): List or numpy array holding the true labels.
        preds (np.array | list): List or numpy array holding the predicted labels.

    Returns:
        The confusion matrix.

    """

    # Making sure that labels is a numpy array
    labels = np.asarray(labels)

    # Making sure that predictions is a numpy array
    preds = np.asarray(preds)

    # Calculating the number of classes
    n_class = np.max(labels)

    # Creating an empty errors matrix
    c_matrix = np.zeros((n_class, n_class))

    # For every label and prediction
    for label, pred in zip(labels, preds):
        # Increments the corresponding cell from the confusion matrix
        c_matrix[label - 1][pred - 1] += 1

    return c_matrix


def normalize(array):
    """Normalizes an input array.

    Args:
        array (np.array): Array to be normalized.

    Returns:
        The normalized version (between 0 and 1) of the input array.

    """

    # Calculates the array mean
    mean = np.mean(array, axis=0)

    # Calculates the array standard deviation
    std = np.std(array, axis=0)

    # Calculates the normalized array
    norm_array = (array - mean) / std

    return norm_array


def opf_accuracy(labels, preds):
    """Calculates the accuracy between true and predicted labels using OPF-style measure.

    Args:
        labels (np.array | list): List or numpy array holding the true labels.
        preds (np.array | list): List or numpy array holding the predicted labels.

    Returns:
        The OPF accuracy measure between 0 and 1.

    """

    # Making sure that labels is a numpy array
    labels = np.asarray(labels)

    # Making sure that predictions is a numpy array
    preds = np.asarray(preds)

    # Calculating the number of classes
    n_class = np.max(labels)

    # Creating an empty errors matrix
    errors = np.zeros((n_class, 2))

    # Gathering the amount of labels per class
    _, counts = np.unique(labels, return_counts=True)

    # For every label and prediction
    for label, pred in zip(labels, preds):
        # If label is different from prediction
        if label != pred:
            # Increments the corresponding cell from the error matrix
            errors[pred - 1][0] += 1

            # Increments the corresponding cell from the error matrix
            errors[label - 1][1] += 1

    # Calculating the float value of the true label errors
    errors[:, 1] /= counts

    # Calculating the float value of the predicted label errors
    errors[:, 0] /= (np.sum(counts) - counts)

    # Calculates the sum of errors per class
    errors = np.sum(errors, axis=1)

    # Calculates the OPF accuracy
    accuracy = 1 - (np.sum(errors) / (2 * n_class))

    return accuracy


def opf_accuracy_per_label(labels, preds):
    """Calculates the accuracy per label between true and predicted labels using OPF-style measure.

    Args:
        labels (np.array | list): List or numpy array holding the true labels.
        preds (np.array | list): List or numpy array holding the predicted labels.

    Returns:
        The OPF accuracy measure per label between 0 and 1.

    """

    # Making sure that labels is a numpy array
    labels = np.asarray(labels)

    # Making sure that predictions is a numpy array
    preds = np.asarray(preds)

    # Calculating the number of classes
    n_class = np.max(labels)

    # Creating an empty errors array
    errors = np.zeros(n_class)

    # Gathering the amount of labels per class
    _, counts = np.unique(labels, return_counts=True)

    # For every label and prediction
    for label, pred in zip(labels, preds):
        # If label is different from prediction
        if label != pred:
            # Increments the corresponding cell from the error array
            errors[label - 1] += 1

    # Calculating the float value of the true label errors
    errors /= counts

    # Calculates the OPF accuracy
    accuracy = 1 - errors

    return accuracy


def pre_compute_distance(data, output, distance='log_squared_euclidean'):
    """Pre-computes a matrix of distances based on an input data.

    Args:
        data (np.array): Array of samples.
        output (str): File to be saved.
        distance (str): Distance metric to be used.

    """

    logger.info('Pre-computing distances ...')

    # Gathering the size of pre-computed matrix
    size = data.shape[0]

    # Creating an matrix of pre-computed distances
    distances = np.zeros((size, size))

    # For every possible size
    for i in range(size):
        # For every possible size
        for j in range(size):
            # Calculates the distance between nodes `i` and `j`
            distances[i][j] = d.DISTANCES[distance](data[i], data[j])

    # Saves the distance matrix to an output
    np.savetxt(output, distances)

    logger.info(f'Distances saved to: {output}.')


def purity(labels, preds):
    """Calculates the purity measure of an unsupervised technique.

    Args:
        labels (np.array | list): List or numpy array holding the true labels.
        preds (np.array | list): List or numpy array holding the assigned labels by the clusters.

    Returns:
        The purity measure.

    """

    # Calculating the confusion matrix
    c_matrix = confusion_matrix(labels, preds)

    # Calculating the purity measure
    purity = np.sum(np.max(c_matrix, axis=0)) / len(labels)

    return purity
