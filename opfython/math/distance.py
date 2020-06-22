import math
import numpy as np

import opfython.utils.constants as c


def bray_curtis_distance(x, y):
    """Calculates the Bray-Curtis Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Bray-Curtis Distance between x and y.

    """

    # Calculating the Bray-Curtis distance for each dimension
    dist = np.einsum('i->', np.fabs(x - y)) / np.einsum('i->', x + y)

    return dist


def canberra_distance(x, y):
    """Calculates the Canberra Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Canberra Distance between x and y.

    """

    # Calculating the Canberra distance for each dimension
    dist = np.fabs(x - y) / (np.fabs(x) + np.fabs(y))

    return np.einsum('i->', dist)


def chebyshev_distance(x, y):
    """Calculates the Chebyshev Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Chebyshev Distance between x and y.

    """

    # Calculates the Chebyshev distance for each dimension
    dist = np.fabs(x - y)

    return np.amax(dist)


def chi_squared_distance(x, y):
    """Calculates the Chi-Squared Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Chi-Squared Distance between x and y.

    """

    # Calculating the Chi-Squared distance for each dimension
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

    # Calculating the Chord distance
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

    # Calculates the Euclidean distance for each dimension
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

    # Calculates the Gaussian distance for each dimension
    dist = (x - y) ** 2

    return math.exp(-gamma * np.einsum('i->', dist) ** 0.5)


def gower_distance(x, y):
    """Calculates the Gower Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Gower Distance between x and y.

    """

    # Calculates the Gower distance for each dimension
    dist = np.fabs(x - y)

    return np.einsum('i->', dist) / x.shape[0]


def kulczynski_distance(x, y):
    """Calculates the Kulczynski Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Kulczynski Distance between x and y.

    """

    # Calculating the Kulczynski distance for each dimension
    dist = np.einsum('i->', np.fabs(x - y)) / np.einsum('i->', np.amin(x, y))

    return dist


def log_euclidean_distance(x, y):
    """Calculates the log-Euclidean Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The log-Euclidean Distance between x and y.

    """

    # Calculates the log-Euclidean distance for each dimension
    dist = euclidean_distance(x, y)

    return c.MAX_ARC_WEIGHT * math.log(dist + 1)


def log_squared_euclidean_distance(x, y):
    """Calculates the log-Squared Euclidean Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Log Squared Euclidean Distance between x and y.

    """

    # Calculates the log-Squared Euclidean distance for each dimension
    dist = squared_euclidean_distance(x, y)

    return c.MAX_ARC_WEIGHT * math.log(dist + 1)


def lorentzian_distance(x, y):
    """Calculates the Lorentzian Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Lorentzian Distance between x and y.

    """

    # Calculates the Lorentzian distance for each dimension
    dist = np.log(1 + np.fabs(x - y))

    return np.einsum('i->', dist)


def manhattan_distance(x, y):
    """Calculates the Manhattan Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Manhattan Distance between x and y.

    """

    # Calculates the Manhattan distance for each dimension
    dist = np.fabs(x - y)

    return np.einsum('i->', dist)


def non_intersection_distance(x, y):
    """Calculates the Non-Intersection Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Non-Intersection Distance between x and y.

    """

    # Calculates the Non-Intersection distance for each dimension
    dist = np.fabs(x - y)

    return np.einsum('i->', dist) * 0.5


def soergel_distance(x, y):
    """Calculates the Soergel Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Soergel Distance between x and y.

    """

    # Calculating the Soergel distance for each dimension
    dist = np.einsum('i->', np.fabs(x - y)) / np.einsum('i->', np.amax(x, y))

    return dist


def squared_chord_distance(x, y):
    """Calculates the Squared Chord Distance, where features must be positive.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Squared Chord Distance between x and y.

    """

    # Calculating the Squared Chord distance for each dimension
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

    # Calculates the Squared Euclidean distance for each dimension
    dist = (x - y) ** 2

    return np.einsum('i->', dist)


# A distances constant dictionary for selecting the desired
# distance metric to be used
DISTANCES = {
    'bray_curtis': bray_curtis_distance,
    'canberra': canberra_distance,
    'chebyshev': chebyshev_distance,
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
