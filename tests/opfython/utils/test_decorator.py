# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import pytest

from opfython.math import distance
from opfython.utils import constants, decorator


def _return_arguments(x, y):
    return x, y


def test_avoid_zero_division():
    @decorator.avoid_zero_division
    def call(x, y):
        return x, y

    x, y = call(1, 1)

    assert x == 1
    assert y == 1


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("storage", ["writable", "readonly", "views", "aliases"])
def test_avoid_zero_division_preserves_borrowed_arrays(dtype, storage):
    base = np.array([0, constants.EPSILON, 1, 0, 2], dtype=dtype)
    if storage == "views":
        x, y = base[:-1], base[1:]
    elif storage == "aliases":
        x = y = base
    else:
        x, y = base.copy(), base[::-1].copy()
        if storage == "readonly":
            x.setflags(write=False)
            y.setflags(write=False)

    original_base, original_x, original_y = base.copy(), x.copy(), y.copy()
    original_flags = x.flags.writeable, y.flags.writeable
    expected_x, expected_y = x.copy(), y.copy()
    expected_x += constants.EPSILON
    expected_y += constants.EPSILON

    result_x, result_y = decorator.avoid_zero_division(_return_arguments)(x, y)

    np.testing.assert_array_equal(base, original_base)
    np.testing.assert_array_equal(x, original_x)
    np.testing.assert_array_equal(y, original_y)
    np.testing.assert_array_equal(result_x, expected_x)
    np.testing.assert_array_equal(result_y, expected_y)
    assert (x.flags.writeable, y.flags.writeable) == original_flags
    assert result_x.dtype == result_y.dtype == dtype
    assert not np.shares_memory(result_x, x)
    assert not np.shares_memory(result_y, y)
    assert not np.shares_memory(result_x, result_y)


@pytest.mark.parametrize("scalar", [0, 1, 0.0, np.float32(0), np.float64(0), np.complex64(0), 0j])
def test_avoid_zero_division_preserves_scalar_operations(scalar):
    expected = scalar
    expected += constants.EPSILON

    x, y = decorator.avoid_zero_division(_return_arguments)(scalar, scalar)

    assert x == y == expected
    assert type(x) is type(y) is type(expected)


def test_avoid_zero_division_preserves_function_metadata():
    wrapped = decorator.avoid_zero_division(_return_arguments)

    assert wrapped.__wrapped__ is _return_arguments
    assert wrapped.__name__ == _return_arguments.__name__
    assert wrapped.__qualname__ == _return_arguments.__qualname__
    assert wrapped.__module__ == _return_arguments.__module__
    assert wrapped.__doc__ == _return_arguments.__doc__


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("aliased", [False, True])
def test_avoid_zero_division_keeps_numba_metrics_callable(dtype, aliased):
    base = np.array([0, 1, 0, 2, 0], dtype=dtype)
    base.setflags(write=False)
    x = base[:-1]
    y = x if aliased else base[1:]
    original = base.copy()
    expected = distance.canberra_distance.__wrapped__(x + constants.EPSILON, y + constants.EPSILON)

    result = distance.canberra_distance(x, y)

    assert result == expected
    assert distance.canberra_distance(x, y) == result
    assert distance.canberra_distance.__wrapped__.signatures
    np.testing.assert_array_equal(base, original)
    assert not base.flags.writeable
