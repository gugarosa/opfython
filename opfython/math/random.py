"""Random number generators."""

import numpy as np


def generate_uniform_random_number(
    low: float = 0.0,
    high: float = 1.0,
    size: int = 1,
) -> np.array:
    """Generate values from a uniform distribution."""

    return np.random.uniform(low, high, size)


def generate_gaussian_random_number(
    mean: float = 0.0,
    variance: float = 1.0,
    size: int = 1,
) -> np.array:
    """Generate values from a Gaussian distribution."""

    return np.random.normal(mean, variance, size)
