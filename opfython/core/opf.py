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

        # Creating an array of paths' costs
        path = np.ones(self.subgraph.n_nodes)

        # Filling the array with the maximum possible value
        path.fill(c.FLOAT_MAX)

        # Creating a Heap of size equals to number of nodes
        h = Heap(self.subgraph.n_nodes)

        # Applying cost zero to the first node's path
        path[0] = 0

        # Marking first node without any predecessor
        self.subgraph.nodes[0].pred = c.NIL

        # Adding first node to the heap
        h.insert(0)

        # A list of prototype nodes
        prototypes = []

        # While the heap is not empty
        while not h.is_empty():
            # Remove a node from the heap
            p = h.remove()

            # Gathers its cost
            self.subgraph.nodes[p].cost = path[p]

            # And also its predecessor
            pred = self.subgraph.nodes[p].pred

            # If the predecessor is not NIL
            if pred != c.NIL:
                # Checks if the label of current node is the same as its predecessor
                if self.subgraph.nodes[p].label != self.subgraph.nodes[pred].label:
                    # If current node is not a prototype
                    if self.subgraph.nodes[p].status != c.PROTOTYPE:
                        # Marks it as a prototype
                        self.subgraph.nodes[p].status = c.PROTOTYPE

                        # Appends current node identifier to the prototype's list
                        prototypes.append(p)

                    # If predecessor node is not a prototype
                    if self.subgraph.nodes[pred].status != c.PROTOTYPE:
                        # Marks it as a protoype
                        self.subgraph.nodes[pred].status = c.PROTOTYPE

                        # Appends predecessor node identifier to the prototype's list
                        prototypes.append(pred)

            # For every possible node
            for q in range(self.subgraph.n_nodes):
                # Checks if the color of current node in the heap is not black
                if h.color[q] != c.BLACK:
                    # If `p` and `q` identifiers are different
                    if p != q:
                        # If it is supposed to use pre-computed distances
                        if self.pre_computed_distance:
                            # Gathers the arc from the distances' matrix
                            weight = self.pre_distances[self.subgraph.nodes[p]
                                                        .idx][self.subgraph.nodes[q].idx]

                        # If distance is supposed to be calculated
                        else:
                            # Calculates the distance
                            weight = d.DISTANCES[self.distance](
                                self.subgraph.nodes[p].features, self.subgraph.nodes[q].features)

                        # If current arc's cost is smaller than the path's cost
                        if weight < path[q]:
                            # Replace the path's cost value
                            path[q] = weight

                            # Marks `q` predecessor node as `p`
                            self.subgraph.nodes[q].pred = p

                            # Updates the arc on the heap
                            h.update(q, weight)

        logger.debug(f'Prototypes: {prototypes}')

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
