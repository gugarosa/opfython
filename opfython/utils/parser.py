import opfython.utils.logging as l

logger = l.get_logger(__name__)

def parse_df(data):
    """Parses a data in OPF file format that was pre-loaded (.csv, .txt or .json).

    Args:
        data (df): Dataframe holding the data in OPF file format.

    Returns:
        Lists holding ids, labels and features parsed from the data.

    """

    logger.debug('Parsing data frame ...')

    # First column should be the idx
    idx = list(data.iloc[:, 0])

    # Second column should hold the labels
    labels = list(data.iloc[:, 1])

    # From third columns, we should have the features
    features = list(data.iloc[:, 2:].values)

    logger.debug(f'Parsed samples: {len(idx)}.')

    return idx, labels, features
