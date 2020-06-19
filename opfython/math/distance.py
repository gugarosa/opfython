import math
import numpy as np

import opfython.utils.constants as c


def bray_curtis_distance(x, y):
    """Calculates the Bray Curtis Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Bray Curtis Distance between x and y.

    """

    # Calculating the auxiliary term
    aux = x + y

    # Replacing negative values with 1
    aux[aux <= 0] = 1

    # Calculating the bray curtis distance for each dimension
    dist = np.fabs(x - y) / aux

    return np.einsum('i->', dist)


def canberra_distance(x, y):
    """Calculates the Canberra Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Canberra Distance between x and y.

    """

    # Calculating the auxiliary term
    aux = np.fabs(x + y)

    # Replacing zero values with 1
    aux[aux == 0] = 1

    # Calculating the canberra distance for each dimension
    dist = np.fabs(x - y) / aux

    return np.einsum('i->', dist)


def chi_squared_distance(x, y):
    """Calculates the Chi-Squared Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Chi-Squared Distance between x and y.

    """

    # Calculating the chi-squared distance for each dimension
    dist = ((x - y) ** 2 / (x + y))

    return np.einsum('i->', dist) * 0.5


def chord_distance(x, y):
    """Calculates the Chord Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Chord Distance between x and y.

    """

    # Calculating the chord distance
    dist = 2 - 2 * np.einsum('i->', x * y) / (np.einsum('i->', x ** 2) * np.einsum('i->', y ** 2))

    return dist ** 0.5


def cosine_distance(x, y):
    """Calculates the Cosine Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The cosine Distance between x and y.

    """

    # Calculating the Cosine distance
    dist = 1 - (np.einsum('i->', x * y) / (np.einsum('i->', x ** 2) ** 0.5 * np.einsum('i->', y ** 2) ** 0.5))

    return dist


def euclidean_distance(x, y):
    """Calculates the Euclidean Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Euclidean Distance between x and y.

    """

    # Calculates the squared euclidean distance for each dimension
    dist = (x - y) ** 2

    return np.einsum('i->', dist) ** 0.5


def gaussian_distance(x, y, gamma=1):
    """Calculates the Gaussian Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Gaussian Distance between x and y.

    """

    # Calculates the squared euclidean distance for each dimension
    dist = (x - y) ** 2

    return math.exp(-gamma * np.einsum('i->', dist) ** 0.5)


def log_euclidean_distance(x, y):
    """Calculates the Log Euclidean Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Log Euclidean Distance between x and y.

    """

    # Calculates the squared euclidean distance for each dimension
    dist = (x - y) ** 2

    return c.MAX_ARC_WEIGHT * math.log(np.einsum('i->', dist) ** 0.5 + 1)


def log_squared_euclidean_distance(x, y):
    """Calculates the Log Squared Euclidean Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Log Squared Euclidean Distance between x and y.

    """

    # Calculates the squared euclidean distance for each dimension
    dist = (x - y) ** 2

    return c.MAX_ARC_WEIGHT * math.log(np.einsum('i->', dist) + 1)


def manhattan_distance(x, y):
    """Calculates the Manhattan Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Manhattan Distance between x and y.

    """

    # Calculates the manhattan distance for each dimension
    dist = np.fabs(x - y)

    return np.einsum('i->', dist)


def squared_chord_distance(x, y):
    """Calculates the Squared Chord Distance, where features must be positive.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Squared Chord Distance between x and y.

    """

    # Calculating the squared chord distance for each dimension
    dist = (x ** 0.5 - y ** 0.5) ** 2

    return np.einsum('i->', dist)


def squared_euclidean_distance(x, y):
    """Calculates the Squared Euclidean Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Squared Euclidean Distance between x and y.

    """

    # Calculates the squared euclidean distance for each dimension
    dist = (x - y) ** 2

    return np.einsum('i->', dist)


# A distances constant dictionary for selecting the desired
# distance metric to be used
DISTANCES = {
    'bray_curtis': bray_curtis_distance,
    'canberra': canberra_distance,
    'chi_squared': chi_squared_distance,
    'chord': chord_distance,
    'cosine': cosine_distance,
    'euclidean': euclidean_distance,
    'gaussian': gaussian_distance,
    'log_euclidean': log_euclidean_distance,
    'log_squared_euclidean': log_squared_euclidean_distance,
    'manhattan': manhattan_distance,
    'squared_chord': squared_chord_distance,
    'squared_euclidean': squared_euclidean_distance
}
