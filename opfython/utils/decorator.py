# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Decorators."""

from collections.abc import Callable
from functools import wraps
from typing import TypeVar

import numpy as np

import opfython.utils.constants as c

T = TypeVar("T")


def avoid_zero_division(f: Callable[..., T]) -> Callable[..., T]:
    """Add epsilon to both arguments without modifying caller-owned storage.

    Args:
        f: Function accepting two numeric scalars or arrays.

    Returns:
        Callable[..., T]: Wrapped function preserving metadata and returning the original function's result.

    """

    @wraps(f)
    def _avoid_zero_division(
        x: np.ndarray | np.number | float | complex,
        y: np.ndarray | np.number | float | complex,
    ) -> T:
        return f(x + c.EPSILON, y + c.EPSILON)

    return _avoid_zero_division
