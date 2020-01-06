import numpy as np


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

    # Gathering the amount of labels per class
    _, counts = np.unique(labels, return_counts=True)

    # Creating an empty errors matrix
    errors = np.zeros((len(counts), 2))

    # For every label and prediction
    for label, pred in zip(labels, preds):
        # If label is different from prediction
        if label != pred:
            # Increments the corresponding cell from the confusion matrix
            errors[pred-1][0] += 1

            # Increments the corresponding cell from the confusion matrix
            errors[label-1][1] += 1

    # Calculating the float value of the true label errors
    errors[:, 1] /= counts

    # Calculating the float value of the predicted label errors
    errors[:, 0] /= (np.sum(counts) - counts)

    # Calculates the sum of errors per class
    errors = np.sum(errors, axis=1)

    # Calculates the OPF accuracy
    accuracy = 1 - (np.sum(errors) / (2 * n_class))

    return accuracy
