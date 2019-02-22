import json as j
import sys

import pandas as pd
from pandas.io.json import json_normalize

import opfython.utils.logging as l

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


def load_json(json_path):
    """Loads a .json file into Pandas's dataframe.
    Please make sure the .json is uniform along all keys and items.

    Args:
        json_path (str): A path to the .json file.

    Returns:
        A Panda's dataframe object (NOT WORKING).

    """

    try:
        # Tries to read .txt file into a dataframe
        json = pd.read_json(json_path, orient='split')

    except FileNotFoundError as e:
        # If file is not found, handle the exception and exit
        logger.error(e)
        raise

    # Expand features nested column
    features = json['features'].apply(pd.Series)

    # Drop old features column
    json = json.drop('features', 1)

    # Concate both into a single dataframe
    json = pd.concat([json, features], axis=1)

    return json


def parse_df(data):
    """Parses a data in OPF file format that was pre-loaded (.csv or .txt).

    Args:
        data (df): A dataframe holding the data in OPF file format.

    Returns:
        Lists holding ids, labels and features parsed from the data.

    """

    # First column should be the ids
    ids = list(data[0])

    # Second column should hold the labels
    labels = list(data[1])

    # From third columns, we should have the features
    features = list(data.iloc[:, 2:].values)

    return ids, labels, features
