import numpy as np

from opfython.math.random import generate_uniform_random_number


def bernoulli_distribution(prob=0.0, size=1):
    """ Generates a bernoulli distribution based on an input probability.

    Args:
        prob (float): probability of distribution.
        size (int): size of array.

    Returns:
        A Bernoulli distribution array.

    """

    # Creating bernoulli array
    bernoulli_array = np.zeros(size)

    # Generating random number
    r = generate_uniform_random_number(0, 1, size)

    # For each dimension
    for i in range(size):
        # If random generated number if smaller than probability
        if (r[i] < prob):
            # Mark as one
            bernoulli_array[i] = 1
        else:
            # If not, mark as zero
            bernoulli_array[i] = 0

    return bernoulli_array
