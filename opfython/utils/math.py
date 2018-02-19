""" A generic math module.
    Some of the mathematical functions used by opytimizer are defined in here.
"""

import numpy as np

from .random import generate_uniform_random_number


def BernoulliDistribution(p):
    r = generate_uniform_random_number(0, 1, size=1)
    if (r < p):
        return 1
    else:
        return 0
