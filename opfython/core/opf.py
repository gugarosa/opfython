import pickle

import numpy as np

import opfython.math.distance as d
import opfython.stream.loader as loader
import opfython.utils.constants as c
import opfython.utils.exception as e
import opfython.utils.logging as l
from opfython.core.heap import Heap

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

        # Distances matrix should be initialized as None
        self.pre_distances = None

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

        logger.debug(
            f'Distance: {self.distance} | Pre-computed distance: {self.pre_computed_distance}.')
        logger.info('Class created.')

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
        if not distances:
            # If not, raises a ValueError
            raise e.ValueError(
                'Pre-computed distances could not been properly loaded')

        return distances

    def find_prototypes(self):
        """Find prototype nodes using the Minimum Spanning Tree (MST) approach.

        """

        logger.debug('Finding prototypes ...')

        #
        path = np.ones(self.subgraph.n_nodes)

        #
        path.fill(c.FLOAT_MAX)

        #
        h = Heap(self.subgraph.n_nodes)

        path[0] = 0
        self.subgraph.nodes[0].pred = c.NIL

        h.insert(0)

        n_proto = 0

        while not h.is_empty():
            p = h.remove()
            self.subgraph.nodes[p].cost = path[p]
            pred = self.subgraph.nodes[p].pred

            # print(path)

            if pred != c.NIL:
                if self.subgraph.nodes[p].label != self.subgraph.nodes[pred].label:
                    if self.subgraph.nodes[p].status != c.PROTOTYPE:
                        self.subgraph.nodes[p].status = c.PROTOTYPE
                        n_proto += 1
                    if self.subgraph.nodes[pred].status != c.PROTOTYPE:
                        self.subgraph.nodes[pred].status = c.PROTOTYPE
                        n_proto += 1

            for q in range(self.subgraph.n_nodes):
                if h.color[q] != c.BLACK:
                    if p != q:
                        if self.pre_computed_distance:
                            weight = self.pre_distances[self.subgraph.nodes[p]
                                                        .idx][self.subgraph.nodes[q].idx]
                        else:
                            weight = d.DISTANCES[self.distance](
                                self.subgraph.nodes[p].features, self.subgraph.nodes[q].features)

                        if weight < path[q]:
                            path[q] = weight
                            self.subgraph.nodes[q].pred = p
                            h.update(q, weight)

        logger.debug('Prototypes found.')

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
