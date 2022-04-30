"""Distance-based metrics.
"""

import math

import numpy as np
from numba import njit

import opfython.utils.constants as c
import opfython.utils.decorator as d


@d.avoid_zero_division
@njit(cache=True)
def additive_symmetric_distance(x: np.array, y: np.array) -> float:
    """Calculates the Additive Symmetric Distance (Symmetric Divergence).

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Additive Symmetric Distance between x and y.

    """

    dist = ((x - y) ** 2 * (x + y)) / (x * y)

    return 2 * np.sum(dist)


@njit(cache=True)
def average_euclidean_distance(x: np.array, y: np.array) -> float:
    """Calculates the Average Euclidean Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Average Euclidean Distance between x and y.

    """

    dist = squared_euclidean_distance(x, y)

    return (dist / x.shape[0]) ** 0.5


@d.avoid_zero_division
@njit(cache=True)
def bhattacharyya_distance(x: np.array, y: np.array) -> float:
    """Calculates the Bhattacharyya Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Bhattacharyya Distance between x and y.

    """

    dist = -math.log(np.sum((x * y) ** 0.5))

    return dist


@d.avoid_zero_division
@njit(cache=True)
def bray_curtis_distance(x: np.array, y: np.array) -> float:
    """Calculates the Bray-Curtis Distance (Sorensen Distance).

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Bray-Curtis Distance between x and y.

    """

    dist = np.sum(np.fabs(x - y)) / np.sum(x + y)

    return dist


@d.avoid_zero_division
@njit(cache=True)
def canberra_distance(x: np.array, y: np.array) -> float:
    """Calculates the Canberra Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Canberra Distance between x and y.

    """

    dist = np.fabs(x - y) / (np.fabs(x) + np.fabs(y))

    return np.sum(dist)


@njit(cache=True)
def chebyshev_distance(x: np.array, y: np.array) -> float:
    """Calculates the Chebyshev Distance (Maximum Value Distance, Lagrange, Chessboard Distance).

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Chebyshev Distance between x and y.

    """

    dist = np.fabs(x - y)

    return np.amax(dist)


@d.avoid_zero_division
@njit(cache=True)
def chi_squared_distance(x: np.array, y: np.array) -> float:
    """Calculates the Chi-Squared Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Chi-Squared Distance between x and y.

    """

    dist = (x - y) ** 2 / (x + y)

    return 0.5 * np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def chord_distance(x: np.array, y: np.array) -> float:
    """Calculates the Chord Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Chord Distance between x and y.

    """

    dist = 2 - 2 * (np.sum(x * y) / (np.sum(x**2) ** 0.5 * np.sum(y**2) ** 0.5))

    return dist**0.5


@d.avoid_zero_division
@njit(cache=True)
def clark_distance(x: np.array, y: np.array) -> float:
    """Calculates the Clark Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Clark Distance between x and y.

    """

    dist = ((x - y) / np.fabs(x + y)) ** 2

    return np.sum(dist) ** 0.5


@d.avoid_zero_division
@njit(cache=True)
def cosine_distance(x: np.array, y: np.array) -> float:
    """Calculates the Cosine Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The cosine Distance between x and y.

    """

    dist = 1 - (np.sum(x * y) / (np.sum(x**2) ** 0.5 * np.sum(y**2) ** 0.5))

    return dist


@d.avoid_zero_division
@njit(cache=True)
def dice_distance(x: np.array, y: np.array) -> float:
    """Calculates the Dice Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Dice Distance between x and y.

    """

    dist = 2 * np.sum(x * y) / (np.sum(x**2) + np.sum(y**2))

    return 1 - dist


@d.avoid_zero_division
@njit(cache=True)
def divergence_distance(x: np.array, y: np.array) -> float:
    """Calculates the Divergence Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Divergence Distance between x and y.

    """

    dist = (x - y) ** 2 / (x + y) ** 2

    return 2 * np.sum(dist)


@njit(cache=True)
def euclidean_distance(x: np.array, y: np.array) -> float:
    """Calculates the Euclidean Distance (L2 Norm, Ruler Distance).

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Euclidean Distance between x and y.

    """

    dist = (x - y) ** 2

    return np.sum(dist) ** 0.5


@njit(cache=True)
def gaussian_distance(x, y, gamma=1):
    """Calculates the Gaussian Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Gaussian Distance between x and y.

    """

    dist = (x - y) ** 2

    return math.exp(-gamma * np.sum(dist) ** 0.5)


@njit(cache=True)
def gower_distance(x: np.array, y: np.array) -> float:
    """Calculates the Gower Distance (Average Manhattan, Mean Character Distance).

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Gower Distance between x and y.

    """

    dist = np.fabs(x - y)

    return np.sum(dist) / x.shape[0]


@njit(cache=True)
def hamming_distance(x: np.array, y: np.array) -> float:
    """Calculates the Hamming Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Hamming Distance between x and y.

    """

    dist = np.count_nonzero(x != y)

    return dist


@d.avoid_zero_division
@njit(cache=True)
def hassanat_distance(x: np.array, y: np.array) -> float:
    """Calculates the Hassanat Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Hassanat Distance between x and y.

    """

    # Creates an empty variable to hold each dimension's
    dist = np.zeros(x.shape[0])

    # Creates a binary mask
    mask = np.minimum(x, y) >= 0

    # Iterates through all dimensions
    for i in range(x.shape[0]):
        if mask[i] is True:
            dist[i] = 1 - (1 + np.minimum(x[i], y[i])) / (1 + np.maximum(x[i], y[i]))

        else:
            dist[i] = 1 - (
                1 + np.minimum(x[i], y[i]) + np.fabs(np.minimum(x[i], y[i]))
            ) / (1 + np.maximum(x[i], y[i]) + np.fabs(np.minimum(x[i], y[i])))

    return np.sum(dist)


@njit(cache=True)
def hellinger_distance(x: np.array, y: np.array) -> float:
    """Calculates the Hellinger Distance (Jeffries-Matusita Distance).

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Hellinger Distance between x and y.

    """

    dist = 2 * (x**0.5 - y**0.5) ** 2

    return np.sum(dist) ** 0.5


@d.avoid_zero_division
def jaccard_distance(x: np.array, y: np.array) -> float:
    """Calculates the Jaccard Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Jaccard Distance between x and y.

    """

    dist = np.sum((x - y) ** 2) / (np.sum(x**2) + np.sum(y**2) - np.sum(x * y))

    return dist


@d.avoid_zero_division
@njit(cache=True)
def jeffreys_distance(x: np.array, y: np.array) -> float:
    """Calculates the Jeffreys Distance (J-Divergence, KL2 Divergence).

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Jeffreys Distance between x and y.

    """

    dist = (x - y) * np.log(x / y)

    return np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def jensen_distance(x: np.array, y: np.array) -> float:
    """Calculates the Jensen Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Jensen Distance between x and y.

    """

    dist = (x * np.log(x) + y * np.log(y)) / 2 - ((x + y) / 2) * np.log((x + y) / 2)

    return 0.5 * np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def jensen_shannon_distance(x: np.array, y: np.array) -> float:
    """Calculates the Jensen-Shannon Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Jensen-Shannon Distance between x and y.

    """

    dist1 = x * np.log((2 * x) / (x + y))
    dist2 = y * np.log((2 * y) / (x + y))

    return 0.5 * (np.sum(dist1) + np.sum(dist2))


@d.avoid_zero_division
@njit(cache=True)
def k_divergence_distance(x: np.array, y: np.array) -> float:
    """Calculates the K Divergence Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The K Divergence Distance between x and y.

    """

    dist = x * np.log((2 * x) / (x + y))

    return np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def kulczynski_distance(x: np.array, y: np.array) -> float:
    """Calculates the Kulczynski Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Kulczynski Distance between x and y.

    """

    dist = np.sum(np.fabs(x - y)) / np.sum(np.minimum(x, y))

    return dist


@d.avoid_zero_division
@njit(cache=True)
def kullback_leibler_distance(x: np.array, y: np.array) -> float:
    """Calculates the Kullback-Leibler Distance (KL Divergence).

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Kullback-Leibler Distance between x and y.

    """

    dist = x * np.log(x / y)

    return np.sum(dist)


@njit(cache=True)
def log_euclidean_distance(x: np.array, y: np.array) -> float:
    """Calculates the log-Euclidean Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The log-Euclidean Distance between x and y.

    """

    dist = euclidean_distance(x, y)

    return c.MAX_ARC_WEIGHT * math.log(dist + 1)


@njit(cache=True)
def log_squared_euclidean_distance(x: np.array, y: np.array) -> float:
    """Calculates the log-Squared Euclidean Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Log Squared Euclidean Distance between x and y.

    """

    dist = squared_euclidean_distance(x, y)

    return c.MAX_ARC_WEIGHT * math.log(dist + 1)


@njit(cache=True)
def lorentzian_distance(x: np.array, y: np.array) -> float:
    """Calculates the Lorentzian Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Lorentzian Distance between x and y.

    """

    dist = np.log(1 + np.fabs(x - y))

    return np.sum(dist)


@njit(cache=True)
def manhattan_distance(x: np.array, y: np.array) -> float:
    """Calculates the Manhattan Distance (L1 Norm, Taxicab Norm, City Block Distance).

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Manhattan Distance between x and y.

    """

    dist = np.fabs(x - y)

    return np.sum(dist)


@njit(cache=True)
def matusita_distance(x: np.array, y: np.array) -> float:
    """Calculates the Matusita Distance, where features must be positive.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Matusita Distance between x and y.

    """

    dist = (x**0.5 - y**0.5) ** 2

    return np.sum(dist) ** 0.5


@d.avoid_zero_division
@njit(cache=True)
def max_symmetric_distance(x: np.array, y: np.array) -> float:
    """Calculates the Max Symmetric Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Max Symmetric Distance between x and y.

    """

    dist1 = (x - y) ** 2 / x
    dist2 = (x - y) ** 2 / y

    return np.maximum(np.sum(dist1), np.sum(dist2))


@d.avoid_zero_division
@njit(cache=True)
def mean_censored_euclidean_distance(x: np.array, y: np.array) -> float:
    """Calculates the Mean Censored Euclidean Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Mean Censored Euclidean Distance between x and y.

    """

    dist = squared_euclidean_distance(x, y)
    diff = np.count_nonzero(x + y != 0)

    return (dist / diff) ** 0.5


@d.avoid_zero_division
@njit(cache=True)
def min_symmetric_distance(x: np.array, y: np.array) -> float:
    """Calculates the Min Symmetric Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Min Symmetric Distance between x and y.

    """

    dist1 = (x - y) ** 2 / x
    dist2 = (x - y) ** 2 / y

    return np.minimum(np.sum(dist1), np.sum(dist2))


@d.avoid_zero_division
@njit(cache=True)
def neyman_distance(x: np.array, y: np.array) -> float:
    """Calculates the Neyman Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Neyman Distance between x and y.

    """

    dist = (x - y) ** 2 / x

    return np.sum(dist)


@njit(cache=True)
def non_intersection_distance(x: np.array, y: np.array) -> float:
    """Calculates the Non-Intersection Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Non-Intersection Distance between x and y.

    """

    dist = np.fabs(x - y)

    return 0.5 * np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def pearson_distance(x: np.array, y: np.array) -> float:
    """Calculates the Pearson Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Pearson Distance between x and y.

    """

    dist = (x - y) ** 2 / y

    return np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def sangvi_distance(x: np.array, y: np.array) -> float:
    """Calculates the Sangvi Distance (Probabilistic Symmetric).

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Sangvi Distance between x and y.

    """

    dist = (x - y) ** 2 / (x + y)

    return 2 * np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def soergel_distance(x: np.array, y: np.array) -> float:
    """Calculates the Soergel Distance (Ruzicka Distance).

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Soergel Distance between x and y.

    """

    dist = np.sum(np.fabs(x - y)) / np.sum(np.maximum(x, y))

    return dist


@d.avoid_zero_division
@njit(cache=True)
def squared_distance(x: np.array, y: np.array) -> float:
    """Calculates the Squared Distance (Triangular Discrimination Distance).

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Squared Distance between x and y.

    """

    dist = (x - y) ** 2 / (x + y)

    return np.sum(dist)


@njit(cache=True)
def squared_chord_distance(x: np.array, y: np.array) -> float:
    """Calculates the Squared Chord Distance, where features must be positive.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Squared Chord Distance between x and y.

    """

    dist = (x**0.5 - y**0.5) ** 2

    return np.sum(dist)


@njit(cache=True)
def squared_euclidean_distance(x: np.array, y: np.array) -> float:
    """Calculates the Squared Euclidean Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Squared Euclidean Distance between x and y.

    """

    dist = (x - y) ** 2

    return np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def statistic_distance(x: np.array, y: np.array) -> float:
    """Calculates the Statistic Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Statistic Distance between x and y.

    """

    m = (x + y) / 2
    dist = (x - m) / m

    return np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def topsoe_distance(x: np.array, y: np.array) -> float:
    """Calculates the Topsoe Distance (Information Statistics).

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Topsoe Distance between x and y.

    """

    dist1 = x * np.log((2 * x) / (x + y))
    dist2 = y * np.log((2 * y) / (x + y))

    return np.sum(dist1) + np.sum(dist2)


@d.avoid_zero_division
@njit(cache=True)
def vicis_symmetric1_distance(x: np.array, y: np.array) -> float:
    """Calculates the Vicis Symmetric 1 Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Vicis Symmetric 1 Distance between x and y.

    """

    dist = (x - y) ** 2 / np.minimum(x, y) ** 2

    return np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def vicis_symmetric2_distance(x: np.array, y: np.array) -> float:
    """Calculates the Vicis Symmetric 2 Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Vicis Symmetric 2 Distance between x and y.

    """

    dist = (x - y) ** 2 / np.minimum(x, y)

    return np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def vicis_symmetric3_distance(x: np.array, y: np.array) -> float:
    """Calculates the Vicis Symmetric 3 Distance.

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Vicis Symmetric 3 Distance between x and y.

    """

    dist = (x - y) ** 2 / np.maximum(x, y)

    return np.sum(dist)


@d.avoid_zero_division
@njit(cache=True)
def vicis_wave_hedges_distance(x: np.array, y: np.array) -> float:
    """Calculates the Vicis-Wave Hedges Distance (Wave-Hedges).

    Args:
        x: N-dimensional array.
        y: N-dimensional array.

    Returns:
        (float): The Vicis-Wave Hedges Distance between x and y.

    """

    dist = np.fabs(x - y) / np.minimum(x, y)

    return np.sum(dist)


# A distances constant dictionary for selecting the desired
# distance metric to be used
DISTANCES = {
    "additive_symmetric": additive_symmetric_distance,
    "average_euclidean": average_euclidean_distance,
    "bhattacharyya": bhattacharyya_distance,
    "bray_curtis": bray_curtis_distance,
    "canberra": canberra_distance,
    "chebyshev": chebyshev_distance,
    "chi_squared": chi_squared_distance,
    "chord": chord_distance,
    "clark": clark_distance,
    "cosine": cosine_distance,
    "dice": dice_distance,
    "divergence": divergence_distance,
    "euclidean": euclidean_distance,
    "gaussian": gaussian_distance,
    "gower": gower_distance,
    "hamming": hamming_distance,
    "hassanat": hassanat_distance,
    "hellinger": hellinger_distance,
    "jaccard": jaccard_distance,
    "jeffreys": jeffreys_distance,
    "jensen": jensen_distance,
    "jensen_shannon": jensen_shannon_distance,
    "k_divergence": k_divergence_distance,
    "kulczynski": kulczynski_distance,
    "kullback_leibler": kullback_leibler_distance,
    "log_euclidean": log_euclidean_distance,
    "log_squared_euclidean": log_squared_euclidean_distance,
    "lorentzian": lorentzian_distance,
    "manhattan": manhattan_distance,
    "matusita": matusita_distance,
    "max_symmetric": max_symmetric_distance,
    "mean_censored_euclidean": mean_censored_euclidean_distance,
    "min_symmetric": min_symmetric_distance,
    "neyman": neyman_distance,
    "non_intersection": non_intersection_distance,
    "pearson": pearson_distance,
    "sangvi": sangvi_distance,
    "soergel": soergel_distance,
    "squared": squared_distance,
    "squared_chord": squared_chord_distance,
    "squared_euclidean": squared_euclidean_distance,
    "statistic": statistic_distance,
    "topsoe": topsoe_distance,
    "vicis_symmetric1": vicis_symmetric1_distance,
    "vicis_symmetric2": vicis_symmetric2_distance,
    "vicis_symmetric3": vicis_symmetric3_distance,
    "vicis_wave_hedges": vicis_wave_hedges_distance,
}
