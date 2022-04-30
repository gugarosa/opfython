"""Data parsing utilities.
"""

import numpy as np

import opfython.utils.exception as e
from opfython.utils import logging

logger = logging.get_logger(__name__)


def parse_loader(data: np.array) -> np.array:
    """Parses data in OPF file format that was pre-loaded (.csv, .txt or .json).

    Args:
        data: Numpy array holding the data in OPF file format.

    Returns:
        (np.array): Arrays holding the features and labels.

    """

    logger.info("Parsing data ...")

    try:
        # From third columns beyond, we should have the features
        X = data[:, 2:]

        # Second column should be the label
        Y = data[:, 1]

        # Calculates the amount of samples per class
        _, counts = np.unique(Y, return_counts=True)

        # If there is only one class
        if len(counts) == 1:
            logger.warning("Parsed data only have a single label.")

        # If there are unsequential labels
        if len(counts) != (np.max(Y) + 1):
            raise e.ValueError(
                "Parsed data should have sequential labels, e.g., 0, 1, ..., n-1"
            )

        logger.info("Data parsed.")

        return X, Y.astype(int)

    except TypeError as error:
        logger.error(error)

        return None, None
