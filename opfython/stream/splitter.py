# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Data splitting utilities."""

import numpy as np
from numpy.typing import ArrayLike

import opfython.utils.exception as e
from opfython.utils.logging import get_logger

logger = get_logger(__name__)


def _split_indexes(
    X: np.ndarray,
    Y: np.ndarray,
    percentage: float,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(random_state)

    if X.shape[0] != Y.shape[0]:
        raise e.SizeError("`X` and `Y` must have the same number of samples.")

    indexes = rng.permutation(X.shape[0])
    halt = int(len(X) * percentage)

    return indexes[:halt], indexes[halt:]


def split(
    X: np.ndarray,
    Y: np.ndarray,
    percentage: float = 0.5,
    random_state: int | None = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split features and labels into two shuffled sets without changing global randomness.

    Args:
        X: Feature array with shape (n_samples, n_features).
        Y: Label array with shape (n_samples,).
        percentage: Fraction whose product with n_samples is truncated to obtain the first split boundary.
        random_state: Legacy NumPy seed, or None to initialize an independent generator from system entropy.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Copied X_1, X_2, Y_1, and Y_2 arrays in shuffled order.

    Raises:
        opfython.utils.exception.SizeError: X and Y have different numbers of samples.
        ValueError: The seed is invalid or the percentage cannot be converted to an integer boundary.

    """

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
    logger.info("Data split.")

    return X_1, X_2, Y_1, Y_2


def split_with_index(
    X: np.ndarray,
    Y: np.ndarray,
    percentage: float = 0.5,
    random_state: int | None = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split features and labels while retaining shuffled indexes and leaving global randomness unchanged.

    Args:
        X: Feature array with shape (n_samples, n_features).
        Y: Label array with shape (n_samples,).
        percentage: Fraction whose product with n_samples is truncated to obtain the first split boundary.
        random_state: Legacy NumPy seed, or None to initialize an independent generator from system entropy.

    Returns:
        tuple: Copied X_1, X_2, Y_1, Y_2 arrays followed by their original row indexes I_1 and I_2.

    Raises:
        opfython.utils.exception.SizeError: X and Y have different numbers of samples.
        ValueError: The seed is invalid or the percentage cannot be converted to an integer boundary.

    """

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
    logger.info("Data split.")

    return X_1, X_2, Y_1, Y_2, first, second


def merge(X_1: ArrayLike, X_2: ArrayLike, Y_1: ArrayLike, Y_2: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Merge two feature and label sets without modifying either input.

    Args:
        X_1: First feature set, promoted to at least two dimensions.
        X_2: Second feature set with matching feature columns, promoted to at least two dimensions.
        Y_1: First set's labels, promoted to at least one dimension.
        Y_2: Second set's labels, promoted to at least one dimension.

    Returns:
        tuple[np.ndarray, np.ndarray]: Stacked features and labels with the first set preceding the second.

    Raises:
        opfython.utils.exception.SizeError: A feature set and its labels have different numbers of samples.
        ValueError: Input shapes cannot be stacked.

    """

    logger.info("Merging data ...")
    X_1, X_2 = np.atleast_2d(X_1, X_2)
    Y_1, Y_2 = np.atleast_1d(Y_1, Y_2)
    X = np.vstack((X_1, X_2))
    Y = np.hstack((Y_1, Y_2))
    if X_1.shape[0] != Y_1.shape[0] or X_2.shape[0] != Y_2.shape[0] or X.shape[0] != Y.shape[0]:
        raise e.SizeError("`X_1`, `Y_1` and `X_2`, `Y_2` must each have matching numbers of samples.")

    logger.debug("X: %s | Y: %s.", X.shape, Y.shape)
    logger.info("Data merged.")

    return X, Y
