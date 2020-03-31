import json as j
import struct

import numpy as np

import opfython.utils.logging as l

logger = l.get_logger(__name__)


def opf2txt(opf_path, output_file=None):
    """Converts a binary OPF file (.dat or .opf) to a .txt file.

    Args:
        opf_path (str): Path to the binary file.
        output_file (str): The path to the output file.

    """

    logger.info(f'Converting file: {opf_path} ...')

    # Defining header format
    header_format = '<iii'

    # Calculating size to be read
    header_size = struct.calcsize(header_format)

    # Opening input opf file
    with open(opf_path, 'rb') as f:
        # Reading binary data and unpacking to desired format
        header_data = struct.unpack(header_format, f.read(header_size))

        # Retrieving number of samples
        n_samples = header_data[0]

        # Retrieving number of features
        n_features = header_data[2]

        # Defining the file format for each subsequent line
        file_format = '<ii'

        # For every possible feature
        for _ in range(n_features):
            # Creates the desired file format
            file_format += 'f'

        # Calculates the size based on the file format
        data_size = struct.calcsize(file_format)

        # Creates an empty list to hold the samples
        samples = []

        # For every possible sample
        for _ in range(n_samples):
            # Reading binary data and unpacking to desired format
            data = struct.unpack(file_format, f.read(data_size))

            # Appending the data to list
            samples.append(data)

    # Closing opf file
    f.close()

    # Checking if there is an output file name
    if not output_file:
        # Defining output file
        output_file = opf_path.split('.')[0] + '.txt'

    # Saving output .txt file
    np.savetxt(output_file, samples, delimiter=' ')

    logger.info(f'File converted to {output_file}.')


def opf2csv(opf_path, output_file=None):
    """Converts a binary OPF file (.dat or .opf) to a .csv file.

    Args:
        opf_path (str): Path to the binary file.
        output_file (str): The path to the output file.

    """

    logger.info(f'Converting file: {opf_path} ...')

    # Defining header format
    header_format = '<iii'

    # Calculating size to be read
    header_size = struct.calcsize(header_format)

    # Opening input opf file
    with open(opf_path, 'rb') as f:
        # Reading binary data and unpacking to desired format
        header_data = struct.unpack(header_format, f.read(header_size))

        # Retrieving number of samples
        n_samples = header_data[0]

        # Retrieving number of features
        n_features = header_data[2]

        # Defining the file format for each subsequent line
        file_format = '<ii'

        # For every possible feature
        for _ in range(n_features):
            # Creates the desired file format
            file_format += 'f'

        # Calculates the size based on the file format
        data_size = struct.calcsize(file_format)

        # Creates an empty list to hold the samples
        samples = []

        # For every possible sample
        for _ in range(n_samples):
            # Reading binary data and unpacking to desired format
            data = struct.unpack(file_format, f.read(data_size))

            # Appending the data to list
            samples.append(data)

    # Closing opf file
    f.close()

    # Checking if there is an output file name
    if not output_file:
        # Defining output file
        output_file = opf_path.split('.')[0] + '.csv'

    # Saving output .txt file
    np.savetxt(output_file, samples, delimiter=',')

    logger.info(f'File converted to {output_file}.')


def opf2json(opf_path, output_file=None):
    """Converts a binary OPF file (.dat or .opf) to a .json file.

    Args:
        opf_path (str): Path to the binary file.
        output_file (str): The path to the output file.

    """

    logger.info(f'Converting file: {opf_path} ...')

    # Defining header format
    header_format = '<iii'

    # Calculating size to be read
    header_size = struct.calcsize(header_format)

    # Opening input opf file
    with open(opf_path, 'rb') as f:
        # Reading binary data and unpacking to desired format
        header_data = struct.unpack(header_format, f.read(header_size))

        # Retrieving number of samples
        n_samples = header_data[0]

        # Retrieving number of features
        n_features = header_data[2]

        # Defining the file format for each subsequent line
        file_format = '<ii'

        # For every possible feature
        for _ in range(n_features):
            # Creates the desired file format
            file_format += 'f'

        # Calculates the size based on the file format
        data_size = struct.calcsize(file_format)

        # Creating a JSON structure
        json = {
            'data': []
        }

        # For every possible sample
        for _ in range(n_samples):
            # Reading binary data and unpacking to desired format
            data = struct.unpack(file_format, f.read(data_size))

            # Appending the data to JSON structure
            json['data'].append({
                'id': data[0],
                'label': data[1],
                'features': list(data[2:])
            })

    # Closing opf file
    f.close()

    # Checking if there is an output file name
    if not output_file:
        # Defining output file
        output_file = opf_path.split('.')[0] + '.json'

    # Opening output file
    with open(output_file, 'w') as f:
        # Dumping JSON to file
        j.dump(json, f)

    logger.info(f'File converted to {output_file}.')
