# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import pytest

from opfython.math import random


def test_random_generators_preserve_shapes():
    assert random.generate_uniform_random_number(0, 1, 5).shape == (5,)
    assert random.generate_gaussian_random_number(0, 1, 3).shape == (3,)


@pytest.mark.parametrize(
    ("generator", "numpy_generator"),
    [
        (random.generate_uniform_random_number, np.random.uniform),
        (random.generate_gaussian_random_number, np.random.normal),
    ],
)
@pytest.mark.parametrize("size", [None, 1, (2, 3)])
def test_random_generators_preserve_numpy_results_and_global_rng(generator, numpy_generator, size):
    state = np.random.get_state()
    try:
        np.random.seed(17)
        actual = generator(1.0, 2.0, size)
        next_actual = np.random.random()

        np.random.seed(17)
        expected = numpy_generator(1.0, 2.0, size)
        next_expected = np.random.random()
    finally:
        np.random.set_state(state)

    assert type(actual) is type(expected)
    np.testing.assert_array_equal(actual, expected)
    assert next_actual == next_expected


@pytest.mark.parametrize(
    ("generator", "first", "second"),
    [
        (random.generate_uniform_random_number, [1.0, 2.0], (1.0, 2.0)),
        (random.generate_gaussian_random_number, [1.0, 2.0], (0.0, 0.0)),
    ],
)
def test_random_generators_accept_broadcast_parameters(generator, first, second):
    state = np.random.get_state()
    try:
        actual = generator(first, second, None)
    finally:
        np.random.set_state(state)

    assert isinstance(actual, np.ndarray)
    np.testing.assert_array_equal(actual, [1.0, 2.0])
