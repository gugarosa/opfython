import pickle

import numpy as np

import opfython.math.distance as d
import opfython.utils.constants as c
import opfython.utils.logging as l
from opfython.core.heap import Heap

logger = l.get_logger(__name__)


class OPF:
    """A basic class to define all common OPF-related methods.

    References:
        J. P. Papa, A. X. Falcão and C. T. N. Suzuki. LibOPF: A library for the design of optimum-path forest classifiers (2015).

    """

    def __init__(self, pre_computed_distance=False):
        """Initialization method.

        Args:
            pre_computed_distance (bool): Whether OPF should use pre-computed distances or not.

        """

        logger.info('Creating class: OPF.')

        # Boolean that indicates whether to use pre-computed distance
        self.pre_computed_distance = pre_computed_distance

        # Initializing an empty subgraph
        self.subgraph = None

        # Distances matrix should be initialized as None
        self.distances = None

        # If OPF should use a pre-computed distance
        if pre_computed_distance:
            # Apply the distances matrix
            self.distances = self._read_distances()

        logger.info('Class created.')

    def _read_distances(self):
        """Reads the distance between nodes from a pre-defined file.
        """

        return 0

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

            if pred is not c.NIL:
                if self.subgraph.nodes[p].label is not self.subgraph.nodes[pred].label:
                    if self.subgraph.nodes[p].status is not c.PROTOTYPE:
                        self.subgraph.nodes[p].status = c.PROTOTYPE
                        n_proto += 1
                    if self.subgraph.nodes[pred].status is not c.PROTOTYPE:
                        self.subgraph.nodes[pred].status = c.PROTOTYPE
                        n_proto += 1

            for q in range(self.subgraph.n_nodes):
                if h.color[q] is not c.BLACK:
                    if p is not q:
                        if self.pre_computed_distance:
                            weight = self.distances[self.subgraph.nodes[p]
                                                    .idx][self.subgraph.nodes[q].idx]
                        else:
                            weight = d.log_squared_euclidean_distance(
                                self.subgraph.nodes[p].features, self.subgraph.nodes[q].features)

                        if weight < path[q]:
                            path[q] = weight
                            self.subgraph.nodes[q].pred = p
                            h.update(q, weight)

        logger.debug('Prototypes found.')

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
