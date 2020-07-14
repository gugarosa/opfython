import numpy as np

from opfython.stream import splitter


def test_split():
    try:
        X = np.ones((5, 2))
        Y = np.ones(6)
        X_1, X_2, Y_1, Y_2 = splitter.split(
            X, Y, percentage=0.5, random_state=1)
    except:
        X = np.ones((6, 2))
        Y = np.ones(6)
        X_1, X_2, Y_1, Y_2 = splitter.split(
            X, Y, percentage=0.5, random_state=1)

    assert X_1.shape == (3, 2)
    assert X_2.shape == (3, 2)
    assert Y_1.shape == (3,)
    assert Y_2.shape == (3,)


def test_split_with_index():
    try:
        X = np.ones((5, 2))
        Y = np.ones(6)
        X_1, X_2, Y_1, Y_2, I_1, I_2 = splitter.split_with_index(
            X, Y, percentage=0.5, random_state=1)
    except:
        X = np.ones((6, 2))
        Y = np.ones(6)
        X_1, X_2, Y_1, Y_2, I_1, I_2 = splitter.split_with_index(
            X, Y, percentage=0.5, random_state=1)

    assert X_1.shape == (3, 2)
    assert X_2.shape == (3, 2)
    assert Y_1.shape == (3,)
    assert Y_2.shape == (3,)
    assert I_1.shape == (3,)
    assert I_2.shape == (3,)


def test_merge():
    try:
        X_1 = np.ones((2, 2))
        Y_1 = np.ones(3)
        X_2 = np.ones((3, 2))
        Y_2 = np.ones(3)
        X, Y = splitter.merge(X_1, X_2, Y_1, Y_2)
    except:
        X_1 = np.ones((3, 2))
        Y_1 = np.ones(3)
        X_2 = np.ones((3, 2))
        Y_2 = np.ones(3)
        X, Y = splitter.merge(X_1, X_2, Y_1, Y_2)

    assert X.shape == (6, 2)
    assert Y.shape == (6,)
