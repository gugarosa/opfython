import pickle

import numpy as np

import opfython.math.distance as d
import opfython.stream.loader as loader
import opfython.utils.constants as c
import opfython.utils.exception as e
import opfython.utils.logging as l
from opfython.core.heap import Heap
from opfython.core.subgraph import Subgraph

logger = l.get_logger(__name__)


class OPF:
    """A basic class to define all common OPF-related methods.

    References:
        J. P. Papa, A. X. Falcão and C. T. N. Suzuki. LibOPF: A library for the design of optimum-path forest classifiers (2015).

    """

    def __init__(self, distance='log_squared_euclidean', pre_computed_distance=None):
        """Initialization method.

        Args:
            distance (str): An indicator of the distance metric to be used.
            pre_computed_distance (str): A pre-computed distance file for feeding into OPF.

        """

        logger.info('Creating class: OPF.')

        # Initializing an empty subgraph
        self.subgraph = None

        # An indicator of the distance metric to be used
        self.distance = distance

        # Gathers the distance function as a property
        self.distance_fn = d.DISTANCES[distance]

        # If OPF should use a pre-computed distance
        if pre_computed_distance:
            # Marks the boolean indicator as True
            self.pre_computed_distance = True

            # Apply the distances matrix
            self.pre_distances = self._read_distances(pre_computed_distance)

        # If OPF should not use a pre-computed distance
        else:
            # Marks the boolean indicator as False
            self.pre_computed_distance = False

            # Marks the pre-distances property as None
            self.pre_distances = None

        logger.debug(
            f'Distance: {self.distance} | Pre-computed distance: {self.pre_computed_distance}.')
        logger.info('Class created.')

    @property
    def subgraph(self):
        """Subgraph: Subgraph's instance.

        """

        return self._subgraph

    @subgraph.setter
    def subgraph(self, subgraph):
        if subgraph is not None:
            if not isinstance(subgraph, Subgraph):
                raise e.TypeError('`subgraph` should be a subgraph')

        self._subgraph = subgraph

    @property
    def distance(self):
        """str: Distance metric to be used.

        """

        return self._distance

    @distance.setter
    def distance(self, distance):
        if distance not in ['bray_curtis', 'canberra', 'chi_squared', 'euclidean', 'gaussian', 'log_euclidean', 'log_squared_euclidean', 'manhattan', 'squared_chi_squared', 'squared_cord', 'squared_euclidean']:
            raise e.TypeError('`distance` should be `bray_curtis`, `canberra`, `chi_squared`, `euclidean`, `gaussian`, `log_euclidean`, `log_squared_euclidean`, `manhattan`, `squared_chi_squared`, `squared_cord` or `squared_euclidean`')

        self._distance = distance

    @property
    def distance_fn(self):
        """bool: Distance function to be used.

        """

        return self._distance_fn

    @distance_fn.setter
    def distance_fn(self, distance_fn):
        if not callable(distance_fn):
            raise e.TypeError('`distance_fn` should be a callable')

        self._distance_fn = distance_fn

    @property
    def pre_computed_distance(self):
        """bool: Whether OPF should use a pre-computed distance or not.

        """

        return self._pre_computed_distance

    @pre_computed_distance.setter
    def pre_computed_distance(self, pre_computed_distance):
        if not isinstance(pre_computed_distance, bool):
            raise e.TypeError('`pre_computed_distance` should be a boolean')

        self._pre_computed_distance = pre_computed_distance

    @property
    def pre_distances(self):
        """np.array: Pre-computed distance matrix.

        """

        return self._pre_distances

    @pre_distances.setter
    def pre_distances(self, pre_distances):
        if pre_distances is not None:
            if not isinstance(pre_distances, np.ndarray):
                raise e.TypeError('`pre_distances` should be a numpy array')

        self._pre_distances = pre_distances

    def _read_distances(self, file_path):
        """Reads the distance between nodes from a pre-defined file.

        Args:
            file_path (str): File to be loaded.

        Returns:
            A matrix with pre-computed distances.

        """

        logger.debug('Running private method: read_distances().')

        # Getting file extension
        extension = file_path.split('.')[-1]

        # Check if extension is .csv
        if extension == 'csv':
            # If yes, call the method that actually loads csv
            distances = loader.load_csv(file_path)

        # Check if extension is .txt
        elif extension == 'txt':
            # If yes, call the method that actually loads txt
            distances = loader.load_txt(file_path)

        # If extension is not recognized
        else:
            # Raises an ArgumentError exception
            raise e.ArgumentError(
                'File extension not recognized. It should be either `.csv` or .txt`')

        # Check if distances have been properly loaded
        if distances is None:
            # If not, raises a ValueError
            raise e.ValueError(
                'Pre-computed distances could not been properly loaded')

        return distances

    def load(self, file_name):
        """Loads the object from a pickle encoding.

        Args:
            file_name (str): Pickle's file path to be loaded.

        """

        logger.info(f'Loading model from file: {file_name} ...')

        # Trying to open the file
        with open(file_name, "rb") as origin_file:
            # Loading model from file
            opf = pickle.load(origin_file)

            # Updating all values
            self.__dict__.update(opf.__dict__)

        logger.info('Model loaded.')

    def save(self, file_name):
        """Saves the object to a pickle encoding.

        Args:
            file_name (str): File's name to be saved.

        """

        logger.info(f'Saving model to file: {file_name} ...')

        # Opening a destination file
        with open(file_name, 'wb') as dest_file:
            # Dumping model to file
            pickle.dump(self, dest_file)

        logger.info('Model saved.')

    def fit(self, X, Y):
        """Fits data in the classifier.

        It should be directly implemented in OPF child classes.

        Args:
            X (np.array): Array of features.
            Y (np.array): Array of labels.

        """

        raise NotImplementedError

    def predict(self, X):
        """Predicts new data using the pre-trained classifier.

        It should be directly implemented in OPF child classes.

        Args:
            X (np.array): Array of features.

        Returns:
            A list of predictions for each record of the data.

        """

        raise NotImplementedError
