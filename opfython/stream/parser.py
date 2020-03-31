import numpy as np

import opfython.utils.exception as e
import opfython.utils.logging as l

logger = l.get_logger(__name__)


def parse_loader(data):
    """Parses data in OPF file format that was pre-loaded (.csv, .txt or .json).

    Args:
        data (np.array): Numpy array holding the data in OPF file format.

    Returns:
        Arrays holding the features and labels.

    """

    logger.info('Parsing data ...')

    # Tries to parse the dataframe
    try:
        # From third columns beyond, we should have the features
        X = data[:, 2:]

        # Second column should be the label
        Y = data[:, 1]

        # Calculates the amount of samples per class
        _, counts = np.unique(Y, return_counts=True)

        # If there is only one class
        if len(counts) < 2:
            # Raises a ValueError
            raise e.ValueError(
                'Parsed data should have at least two distinct labels')

        # If there are unsequential labels
        if len(counts) != np.max(Y):
            # Raises a ValueError
            raise e.ValueError(
                'Parsed data should have sequential labels, e.g., 1, 2, ..., n')

        logger.info(f'Data parsed.')

        return X, Y.astype(int)

    # If dataframe could not be parsed
    except TypeError as error:
        # Logs an error
        logger.error(error)

        return None, None
