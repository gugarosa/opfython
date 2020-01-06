import numpy as np


def accuracy(labels, preds):
    """
    """

    labels = np.asarray(labels)
    preds = np.asarray(preds)

    unique, counts = np.unique(labels, return_counts=True)
    # print(dict(zip(unique, counts)))

    for label, pred in zip(labels, preds):
        if label != pred:
            print(label, pred)
        



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
