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

    def _read_distances(self):
        """
        """
        
        return 0

    def _find_prototypes(self, g):
        """Find prototype nodes using the Minimum Spanning Tree approach.

        Args:
            g (Subgraph): Subgraph to be used.

        """

        #
        path = np.ones(g.n_nodes)

        #
        path.fill(c.FLOAT_MAX)

        #
        h = Heap(g.n_nodes)


        path[0] = 0
        g.nodes[0].pred = c.NIL

        h.insert(0)

        n_proto = 0.0

        while not h.is_empty():
            p = h.remove()
            g.nodes[p].cost = path[p]
            pred = g.nodes[p].pred

            if not pred == -1:
                if not g.nodes[p].label == g.nodes[pred].label:
                    if not g.nodes[p].status == c.PROTOTYPE:
                        g.nodes[p].status = c.PROTOTYPE
                        n_proto += 1
                    if not g.nodes[pred].status == c.PROTOTYPE:
                        g.nodes[pred].status = c.PROTOTYPE
                        n_proto += 1

            for q in range(g.n_nodes):
                if not h.color[q] == c.BLACK:
                    if not p == q:
                        if self.pre_computed_distance:
                            weight = d.log_euclidean_distance(g.nodes[p].features, g.nodes[q].features)
                        else:
                            weight = self.distances[g.nodes[p].idx][g.nodes[q].idx]

                        if weight < path[q]:
                            g.nodes[q].pred = p
                            h.update(q, weight)
