import time

import numpy as np

import opfython.math.distance as distance
import opfython.utils.constants as c
import opfython.utils.exception as e
import opfython.utils.logging as l
from opfython.core.heap import Heap
from opfython.core.opf import OPF
from opfython.subgraphs.knn import KNNSubgraph

logger = l.get_logger(__name__)


class KNNSupervisedOPF(OPF):
    """A KNNSupervisedOPF which implements the supervised version of OPF classifier with a KNN subgraph.

    References:

    """

    def __init__(self, max_k=1, distance='log_squared_euclidean', pre_computed_distance=None):
        """Initialization method.

        Args:
            distance (str): An indicator of the distance metric to be used.
            pre_computed_distance (str): A pre-computed distance file for feeding into OPF.

        """

        logger.info('Overriding class: OPF -> KNNSupervisedOPF.')

        # Override its parent class with the receiving arguments
        super(KNNSupervisedOPF, self).__init__(
            distance=distance, pre_computed_distance=pre_computed_distance)

        # Defining the maximum `k` value for cutting the subgraph
        self.max_k = max_k

        logger.info('Class overrided.')

    def _clustering(self):
        """Clusters the subgraph using the best `k` value.

        Args:
            best_k (int): Best value of k.

        """

        # For every possible node
        for i in range(self.subgraph.n_nodes):
            for i_adj in self.subgraph.nodes[i].adjacency:
                 if self.subgraph.nodes[i].density == self.subgraph.nodes[int(i_adj)].density:
                     insert = True
                     for j_adj in self.subgraph.nodes[i_adj].adjacency:
                        if i == j_adj:
                            insert = False
                        if insert:
                            # Inserts node `i` in the adjacency list of `i_adj`
                            self.subgraph.nodes[i_adj].adjacency.insert(0, i)

                            # Increments the amount of adjacent nodes
                            self.subgraph.nodes[i_adj].n_adjacency += 1



            # # For every possible `k` value
            # for k in range(best_k):
            #     # Gathers node `i` adjacent node
            #     i_adj = int(self.subgraph.nodes[i].adjacency[k])

            #     # If both nodes' density are equal
            #     if self.subgraph.nodes[i].density == self.subgraph.nodes[i_adj].density:
            #         # Turns on the insertion flag
            #         insert = True

            #         # For every possible `k` value
            #         for k in range(best_k):
            #             # Gathers node `j` adjacent node
            #             j_adj = int(self.subgraph.nodes[i_adj].adjacency[k])

            #             # If the nodes are the same
            #             if i == j_adj:
            #                 # Turns off the insertion flag
            #                 insert = False

            #             # If it is supposed to be inserted
            #             if insert:
            #                 # Inserts node `i` in the adjacency list of `i_adj`
            #                 self.subgraph.nodes[i_adj].adjacency.insert(0, i)

            #                 # Increments the amount of adjacent nodes
            #                 self.subgraph.nodes[i_adj].n_adjacency += 1

        # Creating a maximum heap
        h = Heap(size=self.subgraph.n_nodes, policy='max')

        # For every possible node
        for i in range(self.subgraph.n_nodes):
            # Updates the node's cost on the heap
            h.cost[i] = self.subgraph.nodes[i].cost

            # Defines node's `i` predecessor as NIL
            self.subgraph.nodes[i].pred = c.NIL

            # And its root as its same identifier
            self.subgraph.nodes[i].root = i

            # Inserts the node in the heap
            h.insert(i)

        # Resets the `i` and `l` counter
        i = 0

        # While the heap is not empty
        while not h.is_empty():
            # Removes a node
            p = h.remove()

            # Appends its index to the ordered list
            self.subgraph.idx_nodes.append(p)

            i += 1

            # If the node's predecessor is NIL
            if self.subgraph.nodes[p].pred == c.NIL:
                # Updates its cost on the heap
                h.cost[p] = self.subgraph.nodes[p].density

                # Defines its cluster label as `l`
                self.subgraph.nodes[p].predicted_label = self.subgraph.nodes[p].label

            # Apply current node's cost as the heap's cost
            self.subgraph.nodes[p].cost = h.cost[p]

            # Calculates the number of its adjacent nodes
            n_adjacents = self.subgraph.nodes[p].n_adjacency

            # For every possible adjacent node
            for k in range(n_adjacents):
                # Gathers the adjacent identifier
                p_adj = int(self.subgraph.nodes[p].adjacency[k])

                # If its color in the heap is different from `BLACK`
                if h.color[p_adj] != c.BLACK:
                    # Calculates the current cost
                    current_cost = np.minimum(
                        h.cost[p], self.subgraph.nodes[p_adj].density)

                    # If temporary cost is bigger than heap's cost
                    if current_cost > h.cost[p_adj]:
                        # Replaces the temporary cost
                        # current_cost = h.cost[p_adj]

                        # Apply `p_adj` predecessor as `p`
                        self.subgraph.nodes[p_adj].pred = p

                        # Gathers the same root's identifier
                        self.subgraph.nodes[p_adj].root = self.subgraph.nodes[p].root

                        # And its cluster label
                        self.subgraph.nodes[p_adj].cluster_label = self.subgraph.nodes[p].cluster_label

                        h.update(q, current_cost)

    def _learn(self, max_k):

        # Gathers the distance function
        distance_function = distance.DISTANCES[self.distance]

        for k in range(1, max_k + 1):
            self.subgraph.best_k = k

            self.subgraph.create_arcs(k, distance_function, self.pre_computed_distance, self.pre_distances)

            self.subgraph.calculate_pdf(k, distance_function, self.pre_computed_distance, self.pre_distances)

            self._clustering()

            self.subgraph.destroy_arcs()

    def fit(self, X_train, Y_train, X_val, Y_val):
        """Fits data in the classifier.

        Args:
            X (np.array): Array of features.
            Y (np.array): Array of labels.

        """

        logger.info('Fitting classifier ...')

        # Initializing the timer
        start = time.time()

        # Creating a subgraph
        self.subgraph = KNNSubgraph(X_train, Y_val)

        # Checks if it is supposed to use pre-computed distances
        if self.pre_computed_distance:
            # Checks if its size is the same as the subgraph's amount of nodes
            if self.pre_distances.shape[0] != self.subgraph.n_nodes or self.pre_distances.shape[0] != self.subgraph.n_nodes:
                # If not, raises an error
                raise e.BuildError(
                    'Pre-computed distance matrix should have the size of `n_nodes x n_nodes`')

        #
        self._learn(self.max_k)



        # The subgraph has been properly trained
        self.subgraph.trained = True

        # Ending timer
        end = time.time()

        # Calculating training task time
        train_time = end - start

        logger.info('Classifier has been fitted.')
        logger.info(f'Training time: {train_time} seconds.')