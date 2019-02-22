import sys

import opfython.utils.logging as l
import pandas as pd

logger = l.get_logger(__name__)


def load_csv(csv_path):
    """Loads a CSV file into a dataframe object.

    Args:
        csv_path (str): a string holding the .csv's path

    Returns:
        A Panda's dataframe object.

    """

    try:
        # Tries to read .csv file into a dataframe
        csv = pd.read_csv(csv_path, header=None)

    except FileNotFoundError as e:
        # If file is not found, handle the exception and exit
        logger.error(e)
        raise

    return csv


def load_txt(txt_path):
    """Loads a .txt file into Pandas's dataframe.
    Please make sure the .txt is uniform along all rows and columns.

    Args:
        txt_path (str): A path to the .txt file.

    Returns:
        A Panda's dataframe object.

    """

    try:
        # Tries to read .txt file into a dataframe
        txt = pd.read_csv(txt_path, sep=' ', header=None)

    except FileNotFoundError as e:
        # If file is not found, handle the exception and exit
        logger.error(e)
        raise

    return txt