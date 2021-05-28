"""Distance-based metrics.
"""

import math

import numpy as np
from numba import njit

import opfython.utils.constants as c
import opfython.utils.decorator as d


@d.avoid_zero_division
@njit(cache=True)
def additive_symmetric_distance(x, y):
    """Calculates the Additive Symmetric Distance (Symmetric Divergence).

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Additive Symmetric Distance between x and y.

    """

    # Calculates the Additive Symmetric distance for each dimension
    dist = ((x - y) ** 2 * (x + y)) / (x * y)

    return 2 * np.sum(dist)


@njit(cache=True)
def average_euclidean_distance(x, y):
    """Calculates the Average Euclidean Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Average Euclidean Distance between x and y.

    """

    # Calculates the Squared Euclidean distance for each dimension
    dist = squared_euclidean_distance(x, y)

    return (dist / x.shape[0]) ** 0.5


@d.avoid_zero_division
@njit(cache=True)
def bhattacharyya_distance(x, y):
    """Calculates the Bhattacharyya Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Bhattacharyya Distance between x and y.

    """

    # Calculates the Bhattacharyya distance
    dist = -math.log(np.sum((x * y) ** 0.5))

    return dist


@d.avoid_zero_division
@njit(cache=True)
def bray_curtis_distance(x, y):
    """Calculates the Bray-Curtis Distance (Sorensen Distance).

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Bray-Curtis Distance between x and y.

    """

    # Calculates the Bray-Curtis distance
    dist = np.sum(np.fabs(x - y)) / np.sum(x + y)

    return dist


@d.avoid_zero_division
@njit(cache=True)
def canberra_distance(x, y):
    """Calculates the Canberra Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Canberra Distance between x and y.

    """

    # Calculates the Canberra distance for each dimension
    dist = np.fabs(x - y) / (np.fabs(x) + np.fabs(y))

    return np.sum(dist)


@njit(cache=True)
def chebyshev_distance(x, y):
    """Calculates the Chebyshev Distance (Maximum Value Distance, Lagrange, Chessboard Distance).

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Chebyshev Distance between x and y.

    """

    # Calculates the Chebyshev distance for each dimension
    dist = np.fabs(x - y)

    return np.amax(dist)


@d.avoid_zero_division
@njit(cache=True)
def chi_squared_distance(x, y):
    """Calculates the Chi-Squared Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Chi-Squared Distance between x and y.

    """

    # Calculates the Chi-Squared distance for each dimension
    dist = ((x - y) ** 2 / (x + y))

    return 0.5 * np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def chord_distance(x, y):
    """Calculates the Chord Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Chord Distance between x and y.

    """

    # Calculates the Chord distance
    dist = 2 - 2 * (np.sum(x * y) / (np.sum(x ** 2) ** 0.5 * np.sum(y ** 2) ** 0.5))

    return dist ** 0.5


@d.avoid_zero_division
@njit(cache=True)
def clark_distance(x, y):
    """Calculates the Clark Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Clark Distance between x and y.

    """

    # Calculates the Clark distance for each dimension
    dist = ((x - y) / np.fabs(x + y)) ** 2

    return np.sum(dist) ** 0.5


@d.avoid_zero_division
@njit(cache=True)
def cosine_distance(x, y):
    """Calculates the Cosine Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The cosine Distance between x and y.

    """

    # Calculates the Cosine distance
    dist = 1 - (np.sum(x * y) / (np.sum(x ** 2) ** 0.5 * np.sum(y ** 2) ** 0.5))

    return dist


@d.avoid_zero_division
@njit(cache=True)
def dice_distance(x, y):
    """Calculates the Dice Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Dice Distance between x and y.

    """

    # Calculates the Dice distance
    dist = 2 * np.sum(x * y) / (np.sum(x ** 2) + np.sum(y ** 2))

    return 1 - dist


@d.avoid_zero_division
@njit(cache=True)
def divergence_distance(x, y):
    """Calculates the Divergence Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Divergence Distance between x and y.

    """

    # Calculates the Divergence distance for each dimension
    dist = (x - y) ** 2 / (x + y) ** 2

    return 2 * np.sum(dist)


@njit(cache=True)
def euclidean_distance(x, y):
    """Calculates the Euclidean Distance (L2 Norm, Ruler Distance).

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Euclidean Distance between x and y.

    """

    # Calculates the Euclidean distance for each dimension
    dist = (x - y) ** 2

    return np.sum(dist) ** 0.5


@njit(cache=True)
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

    return math.exp(-gamma * np.sum(dist) ** 0.5)


@njit(cache=True)
def gower_distance(x, y):
    """Calculates the Gower Distance (Average Manhattan, Mean Character Distance).

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Gower Distance between x and y.

    """

    # Calculates the Gower distance for each dimension
    dist = np.fabs(x - y)

    return np.sum(dist) / x.shape[0]


@njit(cache=True)
def hamming_distance(x, y):
    """Calculates the Hamming Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Hamming Distance between x and y.

    """

    # Calculates number of occurences `x != y`
    dist = np.count_nonzero(x != y)

    return dist


@d.avoid_zero_division
@njit(cache=True)
def hassanat_distance(x, y):
    """Calculates the Hassanat Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Hassanat Distance between x and y.

    """

    # Creates an empty variable to hold each dimension's
    dist = np.zeros(x.shape[0])

    # Creates a binary mask
    mask = np.minimum(x, y) >= 0

    # Iterates through all dimensions
    for i in range(x.shape[0]):
        # Checks if mask is `True`
        if mask[i] is True:
            # Calculates the Hassanat Distance for true indexes
            dist[i] = 1 - (1 + np.minimum(x[i], y[i])) / (1 + np.maximum(x[i], y[i]))
        else:
            # Calculates the Hassanat Distance for false indexes
            dist[i] = 1 - (1 + np.minimum(x[i], y[i]) + np.fabs(np.minimum(x[i], y[i]))) / (1 + np.maximum(x[i], y[i]) + np.fabs(np.minimum(x[i], y[i])))

    return np.sum(dist)


@njit(cache=True)
def hellinger_distance(x, y):
    """Calculates the Hellinger Distance (Jeffries-Matusita Distance).

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Hellinger Distance between x and y.

    """

    # Calculates the Hellinger distance for each dimension
    dist = 2 * (x ** 0.5 - y ** 0.5) ** 2

    return np.sum(dist) ** 0.5


@d.avoid_zero_division
def jaccard_distance(x, y):
    """Calculates the Jaccard Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Jaccard Distance between x and y.

    """

    # Calculates the Jaccard distance
    dist = np.sum((x - y) ** 2) / (np.sum(x ** 2) + np.sum(y ** 2) - np.sum(x * y))

    return dist


@d.avoid_zero_division
@njit(cache=True)
def jeffreys_distance(x, y):
    """Calculates the Jeffreys Distance (J-Divergence, KL2 Divergence).

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Jeffreys Distance between x and y.

    """

    # Calculates the Jeffreys distance for each dimension
    dist = (x - y) * np.log(x / y)

    return np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def jensen_distance(x, y):
    """Calculates the Jensen Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Jensen Distance between x and y.

    """

    # Calculates the Jensen distance for each dimension
    dist = (x * np.log(x) + y * np.log(y)) / 2 - ((x + y) / 2) * np.log((x + y) / 2)

    return 0.5 * np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def jensen_shannon_distance(x, y):
    """Calculates the Jensen-Shannon Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Jensen-Shannon Distance between x and y.

    """

    # Calculates the first part Jensen-Shannon distance for each dimension
    dist1 = x * np.log((2 * x) / (x + y))

    # Calculates the second part Jensen-Shannon distance for each dimension
    dist2 = y * np.log((2 * y) / (x + y))

    return 0.5 * (np.sum(dist1) + np.sum(dist2))


@d.avoid_zero_division
@njit(cache=True)
def k_divergence_distance(x, y):
    """Calculates the K Divergence Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The K Divergence Distance between x and y.

    """

    # Calculates the K Divergence distance for each dimension
    dist = x * np.log((2 * x) / (x + y))

    return np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def kulczynski_distance(x, y):
    """Calculates the Kulczynski Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Kulczynski Distance between x and y.

    """

    # Calculates the Kulczynski distance
    dist = np.sum(np.fabs(x - y)) / np.sum(np.minimum(x, y))

    return dist


@d.avoid_zero_division
@njit(cache=True)
def kullback_leibler_distance(x, y):
    """Calculates the Kullback-Leibler Distance (KL Divergence).

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Kullback-Leibler Distance between x and y.

    """

    # Calculates the Kullback-Leibler distance for each dimension
    dist = x * np.log(x / y)

    return np.sum(dist)


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
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

    return np.sum(dist)


@njit(cache=True)
def manhattan_distance(x, y):
    """Calculates the Manhattan Distance (L1 Norm, Taxicab Norm, City Block Distance).

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Manhattan Distance between x and y.

    """

    # Calculates the Manhattan distance for each dimension
    dist = np.fabs(x - y)

    return np.sum(dist)


@njit(cache=True)
def matusita_distance(x, y):
    """Calculates the Matusita Distance, where features must be positive.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Matusita Distance between x and y.

    """

    # Calculates the Matusita distance for each dimension
    dist = (x ** 0.5 - y ** 0.5) ** 2

    return np.sum(dist) ** 0.5


@d.avoid_zero_division
@njit(cache=True)
def max_symmetric_distance(x, y):
    """Calculates the Max Symmetric Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Max Symmetric Distance between x and y.

    """

    # Calculates the first partial Max Symmetric distance for each dimension
    dist1 = (x - y) ** 2 / x

    # Calculates the second partial Max Symmetric distance for each dimension
    dist2 = (x - y) ** 2 / y

    return np.maximum(np.sum(dist1), np.sum(dist2))


@d.avoid_zero_division
@njit(cache=True)
def mean_censored_euclidean_distance(x, y):
    """Calculates the Mean Censored Euclidean Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Mean Censored Euclidean Distance between x and y.

    """

    # Calculates the Squared Euclidean distance for each dimension
    dist = squared_euclidean_distance(x, y)

    # Calculates number of occurences `x + y != 0`
    diff = np.count_nonzero(x + y != 0)

    return (dist / diff) ** 0.5


@d.avoid_zero_division
@njit(cache=True)
def min_symmetric_distance(x, y):
    """Calculates the Min Symmetric Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Min Symmetric Distance between x and y.

    """

    # Calculates the first partial Min Symmetric distance for each dimension
    dist1 = (x - y) ** 2 / x

    # Calculates the second partial Min Symmetric distance for each dimension
    dist2 = (x - y) ** 2 / y

    return np.minimum(np.sum(dist1), np.sum(dist2))


@d.avoid_zero_division
@njit(cache=True)
def neyman_distance(x, y):
    """Calculates the Neyman Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Neyman Distance between x and y.

    """

    # Calculates the Neyman distance for each dimension
    dist = (x - y) ** 2 / x

    return np.sum(dist)


@njit(cache=True)
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

    return 0.5 * np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def pearson_distance(x, y):
    """Calculates the Pearson Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Pearson Distance between x and y.

    """

    # Calculates the Pearson distance for each dimension
    dist = (x - y) ** 2 / y

    return np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def sangvi_distance(x, y):
    """Calculates the Sangvi Distance (Probabilistic Symmetric).

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Sangvi Distance between x and y.

    """

    # Calculates the Sangvi distance for each dimension
    dist = (x - y) ** 2 / (x + y)

    return 2 * np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def soergel_distance(x, y):
    """Calculates the Soergel Distance (Ruzicka Distance).

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Soergel Distance between x and y.

    """

    # Calculates the Soergel distance
    dist = np.sum(np.fabs(x - y)) / np.sum(np.maximum(x, y))

    return dist


@d.avoid_zero_division
@njit(cache=True)
def squared_distance(x, y):
    """Calculates the Squared Distance (Triangular Discrimination Distance).

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Squared Distance between x and y.

    """

    # Calculates the Squared distance for each dimension
    dist = (x - y) ** 2 / (x + y)

    return np.sum(dist)


@njit(cache=True)
def squared_chord_distance(x, y):
    """Calculates the Squared Chord Distance, where features must be positive.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Squared Chord Distance between x and y.

    """

    # Calculates the Squared Chord distance for each dimension
    dist = (x ** 0.5 - y ** 0.5) ** 2

    return np.sum(dist)


@njit(cache=True)
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

    return np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def statistic_distance(x, y):
    """Calculates the Statistic Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Statistic Distance between x and y.

    """

    # Calculates the `m` coefficient
    m = (x + y) / 2

    # Calculates the Statistic distance for each dimension
    dist = (x - m) / m

    return np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def topsoe_distance(x, y):
    """Calculates the Topsoe Distance (Information Statistics).

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Topsoe Distance between x and y.

    """

    # Calculates the first part Topsoe distance for each dimension
    dist1 = x * np.log((2 * x) / (x + y))

    # Calculates the second part Topsoe distance for each dimension
    dist2 = y * np.log((2 * y) / (x + y))

    return np.sum(dist1) + np.sum(dist2)


@d.avoid_zero_division
@njit(cache=True)
def vicis_symmetric1_distance(x, y):
    """Calculates the Vicis Symmetric 1 Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Vicis Symmetric 1 Distance between x and y.

    """

    # Calculates the Vicis Symmetric 1 distance for each dimension
    dist = (x - y) ** 2 / np.minimum(x, y) ** 2

    return np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def vicis_symmetric2_distance(x, y):
    """Calculates the Vicis Symmetric 2 Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Vicis Symmetric 2 Distance between x and y.

    """

    # Calculates the Vicis Symmetric 2 distance for each dimension
    dist = (x - y) ** 2 / np.minimum(x, y)

    return np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def vicis_symmetric3_distance(x, y):
    """Calculates the Vicis Symmetric 3 Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Vicis Symmetric 3 Distance between x and y.

    """

    # Calculates the Vicis Symmetric 3 distance for each dimension
    dist = (x - y) ** 2 / np.maximum(x, y)

    return np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def vicis_wave_hedges_distance(x, y):
    """Calculates the Vicis-Wave Hedges Distance (Wave-Hedges).

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Vicis-Wave Hedges Distance between x and y.

    """

    # Calculates the Vicis-Wave Hedges distance for each dimension
    dist = np.fabs(x - y) / np.minimum(x, y)

    return np.sum(dist)


# A distances constant dictionary for selecting the desired
# distance metric to be used
DISTANCES = {
    'additive_symmetric': additive_symmetric_distance,
    'average_euclidean': average_euclidean_distance,
    'bhattacharyya': bhattacharyya_distance,
    'bray_curtis': bray_curtis_distance,
    'canberra': canberra_distance,
    'chebyshev': chebyshev_distance,
    'chi_squared': chi_squared_distance,
    'chord': chord_distance,
    'clark': clark_distance,
    'cosine': cosine_distance,
    'dice': dice_distance,
    'divergence': divergence_distance,
    'euclidean': euclidean_distance,
    'gaussian': gaussian_distance,
    'gower': gower_distance,
    'hamming': hamming_distance,
    'hassanat': hassanat_distance,
    'hellinger': hellinger_distance,
    'jaccard': jaccard_distance,
    'jeffreys': jeffreys_distance,
    'jensen': jensen_distance,
    'jensen_shannon': jensen_shannon_distance,
    'k_divergence': k_divergence_distance,
    'kulczynski': kulczynski_distance,
    'kullback_leibler': kullback_leibler_distance,
    'log_euclidean': log_euclidean_distance,
    'log_squared_euclidean': log_squared_euclidean_distance,
    'lorentzian': lorentzian_distance,
    'manhattan': manhattan_distance,
    'matusita': matusita_distance,
    'max_symmetric': max_symmetric_distance,
    'mean_censored_euclidean': mean_censored_euclidean_distance,
    'min_symmetric': min_symmetric_distance,
    'neyman': neyman_distance,
    'non_intersection': non_intersection_distance,
    'pearson': pearson_distance,
    'sangvi': sangvi_distance,
    'soergel': soergel_distance,
    'squared': squared_distance,
    'squared_chord': squared_chord_distance,
    'squared_euclidean': squared_euclidean_distance,
    'statistic': statistic_distance,
    'topsoe': topsoe_distance,
    'vicis_symmetric1': vicis_symmetric1_distance,
    'vicis_symmetric2': vicis_symmetric2_distance,
    'vicis_symmetric3': vicis_symmetric3_distance,
    'vicis_wave_hedges': vicis_wave_hedges_distance
}
