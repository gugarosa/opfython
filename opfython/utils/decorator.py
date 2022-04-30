"""Decorators.
"""

from functools import wraps

import numpy as np

import opfython.utils.constants as c


def avoid_zero_division(f: callable) -> callable:
    """Adds a minimal value to arguments to avoid zero values.

    Args:
        f: Incoming function.

    Returns:
        (callable): The incoming function with its adjusted arguments.

    """

    @wraps(f)
    def _avoid_zero_division(x: np.array, y: np.array) -> callable:
        """Wraps the function for adjusting its arguments.

        Args:
            x: N-dimensional array.
            y: N-dimensional array.

        Returns:
            (callable): The function itself.

        """

        x += c.EPSILON
        y += c.EPSILON

        return f(x, y)

    return _avoid_zero_division
