import numpy as np
import pytest

from opfython.math import distance


def test_bray_curtis_distance():
    x = np.asarray([1, 2])
    y = np.asarray([3, 4])

    dist = distance.bray_curtis_distance(x, y)

    assert dist == 0.8333333333333333


def test_canberra_distance():
    x = np.asarray([1, 2])
    y = np.asarray([3, 4])

    dist = distance.canberra_distance(x, y)

    assert dist == 0.8333333333333333


def test_chi_squared_distance():
    x = np.asarray([1, 2])
    y = np.asarray([3, 4])

    dist = distance.chi_squared_distance(x, y)

    assert dist == 0.8333333333333333


def test_chord_distance():
    x = np.asarray([1, 2])
    y = np.asarray([3, 4])

    dist = distance.chord_distance(x, y)

    assert dist == 1.3505554412907306


def test_cosine_distance():
    x = np.asarray([1, 2])
    y = np.asarray([3, 4])

    dist = distance.cosine_distance(x, y)

    assert dist == 0.01613008990009257


def test_euclidean_distance():
    x = np.asarray([1, 2])
    y = np.asarray([3, 4])

    dist = distance.euclidean_distance(x, y)

    assert dist == 2.8284271247461903


def test_gaussian_distance():
    x = np.asarray([1, 2])
    y = np.asarray([3, 4])

    dist = distance.gaussian_distance(x, y)

    assert dist == 0.059105746561956225


def test_log_euclidean_distance():
    x = np.asarray([1, 2])
    y = np.asarray([3, 4])

    dist = distance.log_euclidean_distance(x, y)

    assert dist == 134245.4046453526


def test_log_squared_euclidean_distance():
    x = np.asarray([1, 2])
    y = np.asarray([3, 4])

    dist = distance.log_squared_euclidean_distance(x, y)

    assert dist == 219722.45773362197


def test_manhattan_distance():
    x = np.asarray([1, 2])
    y = np.asarray([3, 4])

    dist = distance.manhattan_distance(x, y)

    assert dist == 4


def test_squared_chord_distance():
    x = np.asarray([1, 2])
    y = np.asarray([3, 4])

    dist = distance.squared_chord_distance(x, y)

    assert dist == 0.879044135369865


def test_squared_euclidean_distance():
    x = np.asarray([1, 2])
    y = np.asarray([3, 4])

    dist = distance.squared_euclidean_distance(x, y)

    assert dist == 8
