from opfython.math import random


def test_random_generators_preserve_shapes():
    assert random.generate_uniform_random_number(0, 1, 5).shape == (5,)
    assert random.generate_gaussian_random_number(0, 1, 3).shape == (3,)
