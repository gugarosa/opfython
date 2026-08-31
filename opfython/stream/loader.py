"""Data loading utilities."""

import json

import numpy as np

from opfython.utils import logging

logger = logging.get_logger(__name__)


def load_csv(csv_path: str) -> np.array:
    """Load a comma-separated file into a NumPy array."""

    logger.info("Loading file: %s ...", csv_path)
    try:
        data = np.loadtxt(csv_path, delimiter=",")
    except OSError as error:
        logger.error(error)
        return None

    logger.info("File loaded.")
    return data


def load_txt(txt_path: str) -> np.array:
    """Load a whitespace-separated file into a NumPy array."""

    logger.info("Loading file: %s...", txt_path)
    try:
        data = np.loadtxt(txt_path, delimiter=" ")
    except OSError as error:
        logger.error(error)
        return None

    logger.info("File loaded.")
    return data


def load_json(json_path: str) -> np.array:
    """Load an OPF JSON file into a NumPy array."""

    logger.info("Loading file: %s ...", json_path)
    try:
        with open(json_path, encoding="utf-8") as json_file:
            records = json.load(json_file)["data"]
    except Exception as error:
        logger.error(error)
        return None

    data = np.asarray(
        [[record["id"], record["label"], *record["features"]] for record in records]
    )
    logger.info("File loaded.")
    return data
