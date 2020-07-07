import numpy as np

import opfython.utils.exception as e
import opfython.utils.logging as l

logger = l.get_logger(__name__)


def split(X, Y, percentage=0.5, random_state=1):
    """Splits data into two new sets.

    Args:
        X (np.array): Array of features.
        Y (np.array): Array of labels.
        percentage (float): Percentage of the data that should be in first set.
        random_state (int): An integer that fixes the random seed.

    Returns:
        Two new sets that were created from `X` and `Y`.

    """

    logger.info(f'Splitting data ...')

    # Defining a fixed random seed
    np.random.seed(random_state)

    # Checks if `X` and `Y` have the same size
    if X.shape[0] != Y.shape[0]:
        # If not, raises a SizeError
        raise e.SizeError(
            f'`X` and `Y` should have the same amount of samples')

    # Gathering the indexes
    idx = np.random.permutation(X.shape[0])

    # Calculating where sets should be halted
    halt = int(len(X) * percentage)

    # Gathering two new sets from `X`
    X_1, X_2 = X[idx[:halt], :], X[idx[halt:], :]

    # Gathering two new sets from `Y`
    Y_1, Y_2 = Y[idx[:halt]], Y[idx[halt:]]

    logger.debug(
        f'X_1: {X_1.shape} | X_2: {X_2.shape} | Y_1: {Y_1.shape} | Y_2: {Y_2.shape}.')
    logger.info('Data splitted.')

    return X_1, X_2, Y_1, Y_2


def split_with_index(X, Y, percentage=0.5, random_state=1):
    """Splits data into two new sets.

    Args:
        X (np.array): Array of features.
        Y (np.array): Array of labels.
        percentage (float): Percentage of the data that should be in first set.
        random_state (int): An integer that fixes the random seed.

    Returns:
        Two new sets that were created from `X` and `Y`, along their indexes.

    """

    logger.info(f'Splitting data ...')

    # Defining a fixed random seed
    np.random.seed(random_state)

    # Checks if `X` and `Y` have the same size
    if X.shape[0] != Y.shape[0]:
        # If not, raises a SizeError
        raise e.SizeError(
            f'`X` and `Y` should have the same amount of samples')

    # Gathering the indexes
    idx = np.random.permutation(X.shape[0])

    # Calculating where sets should be halted
    halt = int(len(X) * percentage)

    # Dividing the indexes
    I_1, I_2 = idx[:halt], idx[halt:]

    # Gathering two new sets from `X`
    X_1, X_2 = X[I_1, :], X[I_2, :]

    # Gathering two new sets from `Y`
    Y_1, Y_2 = Y[I_1], Y[I_2]

    logger.debug(
        f'X_1: {X_1.shape} | X_2: {X_2.shape} | Y_1: {Y_1.shape} | Y_2: {Y_2.shape}.')
    logger.info('Data splitted.')

    return X_1, X_2, Y_1, Y_2, I_1, I_2


def merge(X_1, X_2, Y_1, Y_2):
    """Merge two sets into a new set.

    Args:
        X_1 (np.array): First array of features.
        X_2 (np.array): Second array of features.
        Y_1 (np.array): First array of labels.
        Y_2 (np.array): Second array of labels.

    Returns:
        A new merged set that was created from `X_1`, `X_2`, `Y_1` and `Y_2`.

    """

    logger.info(f'Merging data ...')

    # Vertically stacking `X_1` and `X_2`
    X = np.vstack((X_1, X_2))

    # Horizontally stacking `Y_1` and Y_2`
    Y = np.hstack((Y_1, Y_2))

    # Checks if `X` and `Y` have the same size
    if X.shape[0] != Y.shape[0]:
        # If not, raises a SizeError
        raise e.SizeError(
            f'`(X_1, X_2)` and `(Y_1, Y_2)` should have the same amount of samples')

    logger.debug(f'X: {X.shape} | Y: {Y.shape}.')
    logger.info('Data merged.')

    return X, Y
