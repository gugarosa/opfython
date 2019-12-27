import numpy as np
import opfython.utils.loader as loader
import opfython.utils.logging as l

logger = l.get_logger(__name__)

class Dataset():

    def __init__(self, input_file=None):
        # Getting file extension
        extension = input_file.split('.')[-1]

        # Check if extension is .csv or .txt
        if extension == 'csv' or extension == 'txt':
            # If yes, call the method that actually loads the correct extension
            data = loader.load_csv(input_file)

        # Check if extension is .json
        elif extension == 'json':
            # If yes, call the method that actually loads a JSON
            data = loader.load_json(input_file)

