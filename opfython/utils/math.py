""" A generic math module.
    Some of the mathematical functions used by opfython are defined in here.

    # Methods
    bernoulli_distribution(prob, size): generates a bernoulli distribution based on an input probability.
"""

import numpy as np

from .random import generate_uniform_random_number


def bernoulli_distribution(prob, size=1):
    """ Generates a bernoulli distribution based on an input probability.

        # Arguments
            prob: probability of distribution.
            size: size of array.

        # Returns
            bernoulli_array: bernoulli distribution array.
    """
    bernoulli_array = np.zeros(size)
    r = generate_uniform_random_number(0, 1, size)

    for i in range(size):
        if (r[i] < prob):
            bernoulli_array[i] = 1
        else:
            bernoulli_array[i] = 0
            
    return bernoulli_array
