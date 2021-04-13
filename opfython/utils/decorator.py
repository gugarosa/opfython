"""Decorators.
"""

from functools import wraps

import opfython.utils.constants as c


def avoid_zero_division(f):
    """Adds a minimal value to arguments to avoid zero values.

    Args:
        f (callable): Incoming function.

    Returns:
        The incoming function with its adjusted arguments.

    """

    @wraps(f)
    def _avoid_zero_division(x, y):
        """Wraps the function for adjusting its arguments

        Returns:
            The function itself.

        """

        # Adds minimal value to `x` and `y`
        x += c.EPSILON
        y += c.EPSILON

        return f(x, y)

    return _avoid_zero_division
