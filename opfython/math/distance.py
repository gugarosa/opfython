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

    # Gathering the conditional over the auxiliary term
    cond = np.where(aux > 0, aux, 1)

    # Calculating the bray curtis distance for each dimension
    dist = np.fabs(x - y) / cond

    return np.sum(dist)


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

    # Gathering the conditional over the auxiliary term
    cond = np.where(aux > 0, aux, 1)

    # Calculating the canberra distance for each dimension
    dist = np.fabs(x - y) / cond

    return np.sum(dist)


def chi_squared_distance(x, y):
    """Calculates the Chi-Squared Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Chi-Squared Distance between x and y.

    """

    # Calculating the sum on `x` array
    f = np.sum(x)

    # Calculating the sum on `y` array
    g = np.sum(y)

    # Calculating the chi-squared distance for each dimension
    dist = 1 / (x + y + c.EPSILON) * ((x / f - y / g) ** 2)

    return np.sqrt(np.sum(dist))


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

    return np.sqrt(np.sum(dist))


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

    return np.exp(-gamma * np.sqrt(np.sum(dist)))


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

    return c.MAX_ARC_WEIGHT * np.log(np.sqrt(np.sum(dist)) + c.EPSILON)


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

    return c.MAX_ARC_WEIGHT * np.log(np.sum(dist) + 1)


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

    return np.sum(dist)


def squared_chi_squared_distance(x, y):
    """Calculates the Squared Chi-Squared Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Squared Chi-Squared Distance between x and y.

    """

    # Calculating the auxiliary term
    aux = np.fabs(x + y)

    # Gathering the conditional over the auxiliary term
    cond = np.where(aux > 0, aux, 1)

    # Calculating the squared chi-squared distance for each dimension
    dist = ((x - y) ** 2) / cond

    return np.sum(dist)


def squared_cord_distance(x, y):
    """Calculates the Squared Cord Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Squared Cord Distance between x and y.

    """

    # Calculating the first auxiliary term
    aux1 = np.sqrt(x)

    # Calculating the second auxiliary term
    aux2 = np.sqrt(y)

    # Gathering the conditional over the auxiliary term
    cond = np.where((aux1 >= 0) & (aux2 >= 0), aux1 - aux2, 0)

    # Calculating the squared cord distance for each dimension
    dist = cond ** 2

    return np.sum(dist)


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

    return np.sum(dist)


# A distances constant dictionary for selecting the desired
# distance metric to be used
DISTANCES = {
    'bray_curtis': bray_curtis_distance,
    'canberra': canberra_distance,
    'chi_squared': chi_squared_distance,
    'euclidean': euclidean_distance,
    'gaussian': gaussian_distance,
    'log_euclidean': log_euclidean_distance,
    'log_squared_euclidean': log_squared_euclidean_distance,
    'manhattan': manhattan_distance,
    'squared_chi_squared': squared_chi_squared_distance,
    'squared_cord': squared_cord_distance,
    'squared_euclidean': squared_euclidean_distance
}
