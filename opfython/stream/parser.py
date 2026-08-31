"""Data parsing utilities."""

import numpy as np

import opfython.utils.exception as e
from opfython.utils import logging

logger = logging.get_logger(__name__)


def parse_loader(data: np.array) -> np.array:
    """Split OPF-formatted rows into features and integer labels."""

    logger.info("Parsing data ...")
    try:
        X = data[:, 2:]
        Y = data[:, 1]

        _, counts = np.unique(Y, return_counts=True)
        if len(counts) == 1:
            logger.warning("Parsed data only have a single label.")
        if len(counts) != (np.max(Y) + 1):
            raise e.ValueError(
                "Parsed data should have sequential labels, e.g., 0, 1, ..., n-1"
            )

        logger.info("Data parsed.")
        return X, Y.astype(int)
    except TypeError as error:
        logger.error(error)
        return None, None
