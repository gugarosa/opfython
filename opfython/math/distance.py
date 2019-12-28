import numpy as np

import opfython.utils.constants as c


def squared_euclidean_distance(x, y):
    """Calculates the Squared Euclidean Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Squared Euclidean Distance between x and y.

    """
    
    # Calculates the squared euclidean distance for each dimension
    dist = (x - y) ** 2

    return np.sum(dist)


def log_squared_euclidean_distance(x, y):
    """Calculates the Log Squared Euclidean Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Log Squared Euclidean Distance between x and y.

    """
    
    # Calculates the squared euclidean distance for each dimension
    dist = (x - y) ** 2

    return c.MAX_ARC_WEIGHT * np.log(np.sum(dist) + 1)

def euclidean_distance(x, y):
    """Calculates the Euclidean Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Euclidean Distance between x and y.
        
    """
    
    # Calculates the squared euclidean distance for each dimension
    dist = (x - y) ** 2

    return np.sqrt(np.sum(dist))

def log_euclidean_distance(x, y):
    """Calculates the Log Euclidean Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Log Euclidean Distance between x and y.
        
    """
    
    # Calculates the squared euclidean distance for each dimension
    dist = (x - y) ** 2

    return c.MAX_ARC_WEIGHT * np.log(np.sqrt(np.sum(dist)) + c.EPSILON)

def gaussian_distance(x, y, gamma=1):
    """Calculates the Gaussian Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Gaussian Distance between x and y.
        
    """
    
    # Calculates the squared euclidean distance for each dimension
    dist = (x - y) ** 2

    return np.exp(-gamma * np.sqrt(np.sum(dist)))
