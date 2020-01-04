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
        self.g = None

        # Distances matrix should be initialized as None
        self.distances = None

        # If OPF should use a pre-computed distance
        if pre_computed_distance:
            # Apply the distances matrix
            self.distances = self._read_distances()

        logger.info('Class created.')

    def _normalize_features(self):
        """Normalizes the features using a Normal Distribution.

        """

        pass

    def _read_distances(self):
        """Reads the distance between nodes from a pre-defined file.
        """

        return 0

    def _find_prototypes(self, g):
        """Find prototype nodes using the Minimum Spanning Tree approach.

        Args:
            g (Subgraph): Subgraph to be used.

        """

        logger.debug('Finding prototypes ...')

        #
        path = np.ones(g.n_nodes)

        #
        path.fill(c.FLOAT_MAX)

        #
        h = Heap(g.n_nodes)

        path[0] = 0
        g.nodes[0].pred = c.NIL

        h.insert(0)

        n_proto = 0

        while not h.is_empty():
            p = h.remove()
            g.nodes[p].cost = path[p]
            pred = g.nodes[p].pred

            # print(path)

            if pred is not c.NIL:
                if g.nodes[p].label is not g.nodes[pred].label:
                    if g.nodes[p].status is not c.PROTOTYPE:
                        g.nodes[p].status = c.PROTOTYPE
                        n_proto += 1
                    if g.nodes[pred].status is not c.PROTOTYPE:
                        g.nodes[pred].status = c.PROTOTYPE
                        n_proto += 1

            for q in range(g.n_nodes):
                if h.color[q] is not c.BLACK:
                    if p is not q:
                        if self.pre_computed_distance:
                            weight = self.distances[g.nodes[p].idx][g.nodes[q].idx]
                        else:
                            weight = d.log_squared_euclidean_distance(
                                g.nodes[p].features, g.nodes[q].features)

                        if weight < path[q]:
                            path[q] = weight
                            g.nodes[q].pred = p
                            h.update(q, weight)

        logger.debug('Prototypes found.')

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