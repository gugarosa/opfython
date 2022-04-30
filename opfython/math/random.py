"""Random numbers generators.
"""

from typing import Optional

import numpy as np


def generate_uniform_random_number(
    low: Optional[float] = 0.0, high: Optional[float] = 1.0, size: Optional[int] = 1
) -> np.array:
    """Generates a random number or array based on an uniform distribution.

    Args:
        low: Lower interval.
        high: Higher interval.
        size: Size of array.

    Returns:
        (np.array): An uniform random number or array.

    """

    uniform_array = np.random.uniform(low, high, size)

    return uniform_array


def generate_gaussian_random_number(
    mean: Optional[float] = 0.0,
    variance: Optional[float] = 1.0,
    size: Optional[int] = 1,
) -> np.array:
    """Generates a random number or array based on a gaussian distribution.

    Args:
        mean: Gaussian's mean value.
        variance: Gaussian's variance value.
        size: Size of array.

    Returns:
        (np.array): A gaussian random number or array.

    """

    gaussian_array = np.random.normal(mean, variance, size)

    return gaussian_array
