import opfython.utils.logging as l

logger = l.get_logger(__name__)


def parse_array(data):
    """Parses data in OPF file format that was pre-loaded (.csv, .txt or .json).

    Args:
        data (np.array): Numpy array holding the data in OPF file format.

    Returns:
        Arrays holding the features and labels.

    """

    logger.debug('Parsing array ...')

    # Tries to parse the dataframe
    try:
        # Second column should be the label
        Y = data[:, 1]

        # From third columns, we should have the features
        X = data[:, 2:]

        logger.debug(f'Array parsed.')

        return X, Y.astype(int)

    # If dataframe could not be parsed
    except TypeError as e:
        # Logs an error
        logger.error(e)

        return None, None
