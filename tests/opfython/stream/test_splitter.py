import numpy as np
import pytest

from opfython.stream import splitter
from opfython.utils import exception


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
    assert all(
        np.array_equal(left, right)
        for left, right in zip(plain_split, (X_1, X_2, Y_1, Y_2))
    )


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
def test_merge_preserves_numpy_stacking_inputs(
    X_1, X_2, Y_1, Y_2, expected_X, expected_Y
):
    features, labels = splitter.merge(X_1, X_2, Y_1, Y_2)

    np.testing.assert_array_equal(features, expected_X)
    np.testing.assert_array_equal(labels, expected_Y)
