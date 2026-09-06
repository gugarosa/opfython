# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Data parsing utilities."""

import numpy as np
from numpy.typing import ArrayLike

import opfython.utils.exception as e
from opfython.utils.logging import get_logger

logger = get_logger(__name__)


def parse_loader(data: ArrayLike | None) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Split OPF-formatted rows into a feature view and copied integer labels.

    Args:
        data: Array of rows containing sample IDs, zero-based labels, and feature columns.

    Returns:
        tuple: Feature view X and integer labels Y, or (None, None) if parsing encounters a TypeError.

    Raises:
        opfython.utils.exception.ValueError: Labels are not sequential from zero.
        ValueError: Data cannot be reduced or cast as required.
        IndexError: Data does not have the required columns or dimensions.

    """

    logger.info("Parsing data ...")

    try:
        X = data[:, 2:]
        Y = data[:, 1]

        _, counts = np.unique(Y, return_counts=True)
        if len(counts) == 1:
            logger.warning("`n_labels=%s` contains only a single label.", len(counts))
        if len(counts) != (np.max(Y) + 1):
            raise e.ValueError("`Y` must contain sequential labels starting at zero.")

        logger.info("Data parsed.")

        return X, Y.astype(int)
    except TypeError as error:
        logger.error("`data_type=%s` could not be parsed: %s.", type(data).__name__, str(error).rstrip("."))

        return None, None
