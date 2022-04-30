"""Converts OPF binary data to a variety of extensions.
"""

import json as j
import struct
from typing import Optional

import numpy as np

from opfython.utils import logging

logger = logging.get_logger(__name__)


def opf2txt(opf_path: str, output_file: Optional[str] = None) -> None:
    """Converts a binary OPF file (.dat or .opf) to a .txt file.

    Args:
        opf_path: Path to the binary file.
        output_file: The path to the output file.

    """

    logger.info("Converting file: %s ...", opf_path)

    # Defining header format
    header_format = "<iii"

    # Calculating size to be read
    header_size = struct.calcsize(header_format)

    with open(opf_path, "rb") as f:
        # Reading binary data and unpacking to desired format
        header_data = struct.unpack(header_format, f.read(header_size))

        # Retrieving number of samples and features
        n_samples = header_data[0]
        n_features = header_data[2]

        # Defining the file format for each subsequent line
        file_format = "<ii"

        for _ in range(n_features):
            file_format += "f"

        # Calculates the size based on the file format
        data_size = struct.calcsize(file_format)

        # Creates an empty list to hold the samples
        samples = []

        for _ in range(n_samples):
            # Reading binary data and unpacking to desired format
            data = struct.unpack(file_format, f.read(data_size))

            # Appending the data to list
            # Note that we subtract 1 from `labels` column
            samples.append((data[0], data[1] - 1, *data[2:]))

    if not output_file:
        output_file = opf_path.split(".")[0] + ".txt"

    np.savetxt(output_file, samples, delimiter=" ")

    logger.info("File converted to %s.", output_file)


def opf2csv(opf_path: str, output_file: Optional[str] = None) -> None:
    """Converts a binary OPF file (.dat or .opf) to a .csv file.

    Args:
        opf_path: Path to the binary file.
        output_file: The path to the output file.

    """

    logger.info("Converting file: %s ...", opf_path)

    # Defining header format
    header_format = "<iii"

    # Calculating size to be read
    header_size = struct.calcsize(header_format)

    with open(opf_path, "rb") as f:
        # Reading binary data and unpacking to desired format
        header_data = struct.unpack(header_format, f.read(header_size))

        # Retrieving number of samples and features
        n_samples = header_data[0]
        n_features = header_data[2]

        # Defining the file format for each subsequent line
        file_format = "<ii"

        for _ in range(n_features):
            file_format += "f"

        # Calculates the size based on the file format
        data_size = struct.calcsize(file_format)

        # Creates an empty list to hold the samples
        samples = []

        for _ in range(n_samples):
            # Reading binary data and unpacking to desired format
            data = struct.unpack(file_format, f.read(data_size))

            # Appending the data to list
            # Note that we subtract 1 from `labels` column
            samples.append((data[0], data[1] - 1, *data[2:]))

    if not output_file:
        output_file = opf_path.split(".")[0] + ".csv"

    np.savetxt(output_file, samples, delimiter=",")

    logger.info("File converted to %s.", output_file)


def opf2json(opf_path: str, output_file: Optional[str] = None) -> None:
    """Converts a binary OPF file (.dat or .opf) to a .json file.

    Args:
        opf_path: Path to the binary file.
        output_file: The path to the output file.

    """

    logger.info("Converting file: %s ...", opf_path)

    # Defining header format
    header_format = "<iii"

    # Calculating size to be read
    header_size = struct.calcsize(header_format)

    with open(opf_path, "rb") as f:
        # Reading binary data and unpacking to desired format
        header_data = struct.unpack(header_format, f.read(header_size))

        # Retrieving number of samples and features
        n_samples = header_data[0]
        n_features = header_data[2]

        # Defining the file format for each subsequent line
        file_format = "<ii"

        for _ in range(n_features):
            file_format += "f"

        # Calculates the size based on the file format
        data_size = struct.calcsize(file_format)

        # Creating a JSON structure
        json = {"data": []}

        for _ in range(n_samples):
            # Reading binary data and unpacking to desired format
            data = struct.unpack(file_format, f.read(data_size))

            # Appending the data to JSON structure
            json["data"].append(
                {"id": data[0], "label": data[1] - 1, "features": list(data[2:])}
            )

    if not output_file:
        output_file = opf_path.split(".")[0] + ".json"

    with open(output_file, "w") as f:
        j.dump(json, f)

    logger.info("File converted to %s.", output_file)
