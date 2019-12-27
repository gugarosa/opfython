import json as j

import pandas as pd

import opfython.utils.logging as l

logger = l.get_logger(__name__)


def load_csv(csv_path):
    """Loads a CSV file into a dataframe object.

    Args:
        csv_path (str): String holding the .csv's path.

    Returns:
        A Panda's dataframe object.

    """

    logger.debug(f'Loading file: {csv_path} ...')

    # Tries to invoke a function
    try:
        # Reads the .csv file into a dataframe
        csv = pd.read_csv(csv_path, header=None)

    # If the file is not found
    except FileNotFoundError as e:
        # Handles the exception and logs an error
        logger.error(e)

        return None

    logger.debug(f'File loaded.')

    return csv


def load_txt(txt_path):
    """Loads a .txt file into Pandas's dataframe.

    Please make sure the .txt is uniform along all rows and columns.

    Args:
        txt_path (str): A path to the .txt file.

    Returns:
        A Panda's dataframe object.

    """

    logger.debug(f'Loading file: {txt_path} ...')

    # Tries to invoke a function
    try:
        # Reads the .txt file into a dataframe
        txt = pd.read_csv(txt_path, sep=' ', header=None)

    # If the file is not found
    except FileNotFoundError as e:
        # Handles the exception and logs an error
        logger.error(e)

        return None

    logger.debug(f'File loaded.')

    return txt


def load_json(json_path):
    """Loads a .json file into Pandas's dataframe.

    Please make sure the .json is uniform along all keys and items.

    Args:
        json_path (str): Path to the .json file.

    Returns:
        A Panda's dataframe object.

    """

    logger.debug(f'Loading file: {json_path} ...')

    # Tries to invoke a function
    try:
        # Reads the .json file into a dataframe
        json = pd.read_json(json_path, orient='split')

    # If the file is not found
    except Exception as e:
        # Handles the exception and logs an error
        logger.error(e)

        return None

    logger.debug(f'File loaded.')

    # Expand features nested column
    features = json['features'].apply(pd.Series)

    # Drop old features column
    json = json.drop('features', 1)

    # Concate both into a single dataframe
    json = pd.concat([json, features], axis=1)

    return json
