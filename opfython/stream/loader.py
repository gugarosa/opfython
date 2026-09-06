# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Data loading utilities."""

import json
from os import PathLike

import numpy as np

from opfython.utils.logging import get_logger

logger = get_logger(__name__)


def load_csv(csv_path: str | PathLike[str]) -> np.ndarray | None:
    """Load a comma-separated file into a NumPy array.

    Args:
        csv_path: Path to the comma-separated numeric file.

    Returns:
        np.ndarray | None: Numeric data with NumPy's inferred dimensions, or None on an I/O failure.

    Raises:
        ValueError: File contents cannot be parsed as a numeric array.

    """

    logger.info("Loading file: %s ...", csv_path)

    try:
        data = np.loadtxt(csv_path, delimiter=",")
    except OSError as error:
        logger.error("`csv_path=%s` could not be loaded: %s.", csv_path, str(error).rstrip("."))
        return None

    logger.info("File loaded.")

    return data


def load_txt(txt_path: str | PathLike[str]) -> np.ndarray | None:
    """Load a space-delimited file into a NumPy array.

    Args:
        txt_path: Path to the space-delimited numeric file.

    Returns:
        np.ndarray | None: Numeric data with NumPy's inferred dimensions, or None on an I/O failure.

    Raises:
        ValueError: File contents cannot be parsed as a numeric array.

    """

    logger.info("Loading file: %s...", txt_path)

    try:
        data = np.loadtxt(txt_path, delimiter=" ")
    except OSError as error:
        logger.error("`txt_path=%s` could not be loaded: %s.", txt_path, str(error).rstrip("."))
        return None

    logger.info("File loaded.")

    return data


def load_json(json_path: str | PathLike[str]) -> np.ndarray | None:
    """Load an OPF JSON file into rows of sample IDs, labels, and features.

    Args:
        json_path: Path to a UTF-8 JSON object containing records under the data key.

    Returns:
        np.ndarray | None: NumPy-inferred rows, or None on I/O, decoding, or top-level schema failure.

    Raises:
        KeyError: A record lacks an id, label, or features key.
        TypeError: Records or their features cannot be iterated or indexed as required.
        ValueError: Record rows cannot form a NumPy array.

    """

    logger.info("Loading file: %s ...", json_path)

    try:
        with open(json_path, encoding="utf-8") as json_file:
            records = json.load(json_file)["data"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        logger.error("`json_path=%s` could not be loaded: %s.", json_path, str(error).rstrip("."))
        return None

    data = np.asarray([[record["id"], record["label"], *record["features"]] for record in records])
    logger.info("File loaded.")

    return data
