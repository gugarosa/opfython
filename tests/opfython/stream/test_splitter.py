# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import pytest

from opfython.stream import splitter
from opfython.utils import exception


@pytest.fixture
def global_rng_state():
    original = np.random.get_state()
    np.random.seed(321)
    np.random.normal()
    try:
        yield np.random.get_state()
    finally:
        np.random.set_state(original)


def _assert_rng_state_unchanged(before):
    after = np.random.get_state()
    assert after[0] == before[0]
    np.testing.assert_array_equal(after[1], before[1])
    assert after[2:] == before[2:]


def test_split_and_merge():
    X = np.arange(12).reshape(6, 2)
    Y = np.arange(6)

    X_1, X_2, Y_1, Y_2, I_1, I_2 = splitter.split_with_index(
        X,
        Y,
        percentage=0.5,
        random_state=1,
    )

    assert X_1.shape == X_2.shape == (3, 2)
    assert Y_1.shape == Y_2.shape == (3,)
    assert np.array_equal(X_1, X[I_1])
    assert np.array_equal(X_2, X[I_2])

    X_merged, Y_merged = splitter.merge(X_1, X_2, Y_1, Y_2)
    assert X_merged.shape == (6, 2)
    assert Y_merged.shape == (6,)

    plain_split = splitter.split(X, Y, percentage=0.5, random_state=1)
    assert all(np.array_equal(left, right) for left, right in zip(plain_split, (X_1, X_2, Y_1, Y_2)))


def test_split_rejects_mismatched_lengths():
    with pytest.raises(exception.SizeError):
        splitter.split(np.ones((5, 2)), np.ones(6))


def test_merge_rejects_mismatched_lengths():
    with pytest.raises(exception.SizeError):
        splitter.merge(
            np.ones((2, 2)),
            np.ones((3, 2)),
            np.ones(3),
            np.ones(3),
        )


def test_merge_rejects_mismatches_even_when_combined_lengths_match():
    with pytest.raises(exception.SizeError):
        splitter.merge(
            np.ones((2, 2)),
            np.ones((3, 2)),
            np.zeros(3, dtype=int),
            np.zeros(2, dtype=int),
        )


@pytest.mark.parametrize(
    ("X_1", "X_2", "Y_1", "Y_2", "expected_X", "expected_Y"),
    [
        ([[0], [1]], [[2]], [0, 1], [1], [[0], [1], [2]], [0, 1, 1]),
        ([0, 1], [2, 3], [0], [1], [[0, 1], [2, 3]], [0, 1]),
        (0, 1, 0, 1, [[0], [1]], [0, 1]),
    ],
)
def test_merge_preserves_numpy_stacking_inputs(X_1, X_2, Y_1, Y_2, expected_X, expected_Y):
    features, labels = splitter.merge(X_1, X_2, Y_1, Y_2)

    np.testing.assert_array_equal(features, expected_X)
    np.testing.assert_array_equal(labels, expected_Y)


@pytest.mark.parametrize("function", [splitter.split, splitter.split_with_index])
@pytest.mark.parametrize(
    ("seed", "permutation"),
    [
        (0, [6, 2, 1, 7, 3, 0, 5, 4]),
        (1, [7, 2, 1, 6, 0, 4, 3, 5]),
        (42, [1, 5, 0, 7, 2, 4, 3, 6]),
        (4294967295, [1, 7, 6, 0, 5, 4, 2, 3]),
    ],
)
def test_split_preserves_historical_seeded_order(function, seed, permutation, global_rng_state):
    X, Y = np.arange(16).reshape(8, 2), np.arange(8)

    result = function(X, Y, percentage=0.375, random_state=seed)

    np.testing.assert_array_equal(result[0], X[permutation[:3]])
    np.testing.assert_array_equal(result[1], X[permutation[3:]])
    np.testing.assert_array_equal(result[2], Y[permutation[:3]])
    np.testing.assert_array_equal(result[3], Y[permutation[3:]])
    if function is splitter.split_with_index:
        np.testing.assert_array_equal(result[4], permutation[:3])
        np.testing.assert_array_equal(result[5], permutation[3:])

    _assert_rng_state_unchanged(global_rng_state)


@pytest.mark.parametrize("function", [splitter.split, splitter.split_with_index])
def test_split_with_none_seed_does_not_change_global_state(function, global_rng_state):
    X, Y = np.arange(16).reshape(8, 2), np.arange(8)

    result = function(X, Y, random_state=None)

    assert result[0].shape == result[1].shape == (4, 2)
    np.testing.assert_array_equal(np.sort(np.concatenate(result[2:4])), Y)
    np.testing.assert_array_equal(result[0], X[result[2]])
    np.testing.assert_array_equal(result[1], X[result[3]])
    if function is splitter.split_with_index:
        np.testing.assert_array_equal(result[4], result[2])
        np.testing.assert_array_equal(result[5], result[3])

    _assert_rng_state_unchanged(global_rng_state)


@pytest.mark.parametrize("function", [splitter.split, splitter.split_with_index])
@pytest.mark.parametrize(
    ("n_labels", "percentage", "seed", "error"),
    [
        (7, 0.5, 1, exception.SizeError),
        (7, 0.5, None, exception.SizeError),
        (8, float("nan"), 1, ValueError),
        (8, 0.5, -1, ValueError),
        (8, 0.5, 2**32, ValueError),
        (8, 0.5, "seed", TypeError),
    ],
)
def test_split_failure_does_not_change_global_state(function, n_labels, percentage, seed, error, global_rng_state):
    with pytest.raises(error):
        function(np.arange(16).reshape(8, 2), np.arange(n_labels), percentage=percentage, random_state=seed)

    _assert_rng_state_unchanged(global_rng_state)


@pytest.mark.parametrize("percentage", [-0.25, 0, 0.3, 1, 1.25])
def test_split_preserves_truncation_and_index_slicing(percentage):
    X, Y = np.arange(16).reshape(8, 2), np.arange(8)
    expected = np.array([7, 2, 1, 6, 0, 4, 3, 5])
    halt = int(8 * percentage)

    result = splitter.split_with_index(X, Y, percentage=percentage)

    np.testing.assert_array_equal(result[4], expected[:halt])
    np.testing.assert_array_equal(result[5], expected[halt:])
