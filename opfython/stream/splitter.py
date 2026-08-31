"""Data splitting utilities."""

from typing import Tuple

import numpy as np

import opfython.utils.exception as e
from opfython.utils import logging

logger = logging.get_logger(__name__)


def _split_indexes(
    X: np.array,
    Y: np.array,
    percentage: float,
    random_state: int,
) -> Tuple[np.array, np.array]:
    np.random.seed(random_state)
    if X.shape[0] != Y.shape[0]:
        raise e.SizeError("`X` and `Y` should have the same amount of samples")

    indexes = np.random.permutation(X.shape[0])
    halt = int(len(X) * percentage)
    return indexes[:halt], indexes[halt:]


def split(
    X: np.array,
    Y: np.array,
    percentage: float = 0.5,
    random_state: int = 1,
) -> Tuple[np.array, np.array, np.array, np.array]:
    """Split features and labels into two shuffled sets."""

    logger.info("Splitting data ...")
    first, second = _split_indexes(X, Y, percentage, random_state)
    X_1, X_2, Y_1, Y_2 = X[first], X[second], Y[first], Y[second]
    logger.debug(
        "X_1: %s | X_2: %s | Y_1: %s | Y_2: %s.",
        X_1.shape,
        X_2.shape,
        Y_1.shape,
        Y_2.shape,
    )
    logger.info("Data splitted.")
    return X_1, X_2, Y_1, Y_2


def split_with_index(
    X: np.array,
    Y: np.array,
    percentage: float = 0.5,
    random_state: int = 1,
) -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    """Split features and labels while retaining the shuffled indexes."""

    logger.info("Splitting data ...")
    first, second = _split_indexes(X, Y, percentage, random_state)
    X_1, X_2, Y_1, Y_2 = X[first], X[second], Y[first], Y[second]
    logger.debug(
        "X_1: %s| X_2: %s | Y_1: %s | Y_2: %s.",
        X_1.shape,
        X_2.shape,
        Y_1.shape,
        Y_2.shape,
    )
    logger.info("Data splitted.")
    return X_1, X_2, Y_1, Y_2, first, second


def merge(
    X_1: np.array, X_2: np.array, Y_1: np.array, Y_2: np.array
) -> Tuple[np.array, np.array]:
    """Merge two feature and label sets."""

    logger.info("Merging data ...")
    X = np.vstack((X_1, X_2))
    Y = np.hstack((Y_1, Y_2))
    if X.shape[0] != Y.shape[0]:
        raise e.SizeError(
            "`(X_1, X_2)` and `(Y_1, Y_2)` should have the same amount of samples"
        )

    logger.debug("X: %s | Y: %s.", X.shape, Y.shape)
    logger.info("Data merged.")
    return X, Y
