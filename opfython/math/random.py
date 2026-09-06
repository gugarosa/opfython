# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Random number generators."""

import numpy as np
from numpy.typing import ArrayLike


def generate_uniform_random_number(
    low: ArrayLike = 0.0,
    high: ArrayLike = 1.0,
    size: int | tuple[int, ...] | None = 1,
) -> np.ndarray | float:
    """Generate values from a uniform distribution using NumPy's global random state.

    Args:
        low: Lower bound, inclusive, broadcastable to the output shape.
        high: Upper bound, nominally exclusive, broadcastable to the output shape.
        size: Output shape or None to infer the shape from the bounds.

    Returns:
        np.ndarray | float: Uniform samples, with a scalar returned only for scalar bounds and size=None.

    Raises:
        ValueError: The bounds or size are invalid or cannot be broadcast together.

    """

    return np.random.uniform(low, high, size)


def generate_gaussian_random_number(
    mean: ArrayLike = 0.0,
    variance: ArrayLike = 1.0,
    size: int | tuple[int, ...] | None = 1,
) -> np.ndarray | float:
    """Generate values from a Gaussian distribution using NumPy's global random state.

    Args:
        mean: Distribution mean, broadcastable to the output shape.
        variance: Nonnegative standard deviation passed as NumPy's scale, despite this legacy parameter name.
        size: Output shape or None to infer the shape from mean and variance.

    Returns:
        np.ndarray | float: Gaussian samples, with a scalar returned only for scalar parameters and size=None.

    Raises:
        ValueError: The scale is negative or the parameter shapes and size cannot be broadcast together.

    """

    return np.random.normal(mean, variance, size)
