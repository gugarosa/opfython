import numpy as np

import opfython.utils.constants as c
import opfython.utils.logging as l
from opfython.core.heap import Heap

logger = l.get_logger(__name__)

class OPF:
    """A basic class to define all common OPF-related methods.

    """

    def __init__(self, pre_computed_distance=False):
        """Initialization method.
        
        """

        logger.info('Creating class: OPF.')

        # Boolean that indicates whether to use pre-computed distance
        self.pre_computed_distance = pre_computed_distance

        logger.info('Class created.')

    def _find_prototypes(self, subgraph):
        """
        """

        #
        path = np.ones(subgraph.n_nodes)

        #
        h = Heap(subgraph.n_nodes)


        path[0] = 0
        subgraph.nodes[0].pred = -1

        h.insert(0)

        n_proto = 0.0

        while not h.is_empty():
            p = h.remove()
            subgraph.nodes[p].cost = path[p]
            pred = subgraph.nodes[p].pred

            if not pred == -1:
                if not subgraph.nodes[p].true_label == subgraph.nodes[pred].true_label:
                    if not subgraph.nodes[p].status == 1:
                        subgraph.nodes[p].status = 1
                        n_proto += 1
                    if not subgraph.nodes[pred].status == 1:
                        subgraph.nodes[pred].status = 1
                        n_proto += 1

            for i in range(subgraph.n_nodes):
                if not h.color[i] == c.BLACK:
                    if not p == i:
                        weight = 1
                        if weight < path[i]:
                            subgraph.nodes[i].pred = p
                            h.update(i, weight)
