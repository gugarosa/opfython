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


class UnsupervisedOPF(OPF):
    """An UnsupervisedOPF which implements the unsupervised version of OPF classifier.

    References:
        

    """

    def __init__(self, min_k=1, max_k=1, distance='log_squared_euclidean', pre_computed_distance=None):
        """Initialization method.

        Args:
            distance (str): An indicator of the distance metric to be used.
            pre_computed_distance (str): A pre-computed distance file for feeding into OPF.

        """

        logger.info('Overriding class: OPF -> UnsupervisedOPF.')

        # Override its parent class with the receiving arguments
        super(UnsupervisedOPF, self).__init__(
            distance=distance, pre_computed_distance=pre_computed_distance)

        # Defining the minimum `k` value for cutting the subgraph
        self.min_k = min_k

        # Defining the maximum `k` value for cutting the subgraph
        self.max_k = max_k

        logger.info('Class overrided.')

    @property
    def min_k(self):
        """int: Minimum `k` value for cutting the subgraph.

        """

        return self._min_k

    @min_k.setter
    def min_k(self, min_k):
        if not isinstance(min_k, int):
            raise e.TypeError('`min_k` should be an integer')
        if min_k < 1:
            raise e.ValueError('`min_k` should be >= 1')

        self._min_k = min_k

    @property
    def max_k(self):
        """int: Maximum `k` value for cutting the subgraph.

        """

        return self._max_k

    @max_k.setter
    def max_k(self, max_k):
        if not isinstance(max_k, int):
            raise e.TypeError('`max_k` should be an integer')
        if max_k < 1:
            raise e.ValueError('`max_k` should be >= 1')

        self._max_k = max_k

    def _clustering(self, best_k):
        """Clusters the subgraph using the best `k` value.

        Args:
            best_k (int): Best value of k.

        """

        # For every possible node
        for i in range(self.subgraph.n_nodes):
            # For every possible `k` value
            for k in range(best_k):
                # Gathers node `i` adjacent node
                i_adj = int(self.subgraph.nodes[i].adjacency[k])

                # If both nodes' density are equal
                if self.subgraph.nodes[i].density == self.subgraph.nodes[i_adj].density:
                    # Turns on the insertion flag
                    insert = True

                    # For every possible `k` value
                    for k in range(best_k):
                        # Gathers node `j` adjacent node
                        j_adj = int(self.subgraph.nodes[i_adj].adjacency[k])

                        # If the nodes are the same
                        if i == j_adj:
                            # Turns off the insertion flag
                            insert = False
                        
                        # If it is supposed to be inserted
                        if insert:
                            # Inserts node `i` in the adjacency list of `i_adj`
                            self.subgraph.nodes[i_adj].adjacency.insert(0, i)

                            # Increments the amount of adjacent nodes
                            self.subgraph.nodes[i_adj].n_adjacency += 1

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
        l = 0

        # While the heap is not empty
        while not h.is_empty():
            # Removes a node
            p = h.remove()

            # Appends its index to the ordered list
            self.subgraph.idx_nodes.append(p)

            # If the node's predecessor is NIL
            if self.subgraph.nodes[p].pred == c.NIL:
                # Updates its cost on the heap
                h.cost[p] = self.subgraph.nodes[p].density

                # Defines its cluster label as `l`
                self.subgraph.nodes[p].cluster_label = l

                # Increments the cluster identifier
                l += 1

            # Apply current node's cost as the heap's cost
            self.subgraph.nodes[p].cost = h.cost[p]

            # Calculates the number of its adjacent nodes
            n_adjacents = self.subgraph.nodes[p].n_adjacency + best_k

            # For every possible adjacent node
            for k in range(n_adjacents):
                # Gathers the adjacent identifier
                p_adj = int(self.subgraph.nodes[p].adjacency[k])

                # If its color in the heap is different from `BLACK`
                if h.color[p_adj] != c.BLACK:
                    # Calculates a temporary cost
                    temp_cost = np.minimum(h.cost[p], self.subgraph.nodes[p_adj].density)

                    # If temporary cost is bigger than heap's cost
                    if temp_cost > h.cost[p_adj]:
                        # Replaces the temporary cost
                        temp_cost = h.cost[p_adj]

                        # Apply `p_adj` predecessor as `p`
                        self.subgraph.nodes[p_adj].pred = p

                        # Gathers the same root's identifier
                        self.subgraph.nodes[p_adj].root = self.subgraph.nodes[p].root

                        # And its cluster label
                        self.subgraph.nodes[p_adj].cluster_label = self.subgraph.nodes[p].cluster_label

        # The final number of clusters will be equal to `l`
        self.subgraph.n_clusters = l

    def _normalized_cut(self, best_k, distance_function):
        """
        """

        internal_cluster = np.zeros(self.subgraph.n_clusters)
        external_cluster = np.zeros(self.subgraph.n_clusters)
        cut = 0

        for i in range(self.subgraph.n_nodes):
            n_adjacents = self.subgraph.nodes[i].n_adjacency + best_k
            for k in range(n_adjacents):
                i_adj = int(self.subgraph.nodes[i].adjacency[k])
                if self.pre_computed_distance:
                    distance = self.pre_distances[self.subgraph.nodes[i].idx][self.subgraph.nodes[i_adj].idx]
                else:
                    distance = distance_function(self.subgraph.nodes[i].features, self.subgraph.nodes[i_adj].features)
                if distance > 0:
                    if self.subgraph.nodes[i].cluster_label == self.subgraph.nodes[i_adj].cluster_label:
                        internal_cluster[self.subgraph.nodes[i].cluster_label] += 1 / distance
                    else:
                        external_cluster[self.subgraph.nodes[i].cluster_label] += 1 / distance

        for l in range(self.subgraph.n_clusters):
            if internal_cluster[l] + external_cluster[l] > 0:
                cut += external_cluster[l] / (internal_cluster[l] + external_cluster[l])

        return cut

            
    def _best_minimum_cut(self, min_k, max_k):
        """Performs a minimum cut on the subgraph using the best `k` value.

        Args:
            min_k (int): Minimum value of k.
            max_k (int): Maximum value of k.

        """

        # Gathers the distance function
        distance_function = distance.DISTANCES[self.distance]

        # Calculates the maximum possible distances
        max_distances = self.subgraph.create_arcs(max_k, distance_function, self.pre_computed_distance, self.pre_distances)
        
        # Initialize the minimum cut as maximum possible value
        min_cut = c.FLOAT_MAX
        
        # For every possible value of `k`
        for k in range(min_k, max_k + 1):
            # If minimum cut is different than zero
            if min_cut != 0:
                # Gathers the subgraph's density
                self.subgraph.density = max_distances[k - 1]

                # Gathers the subgraph's best `k` value
                self.subgraph.best_k = k

                # Calculates the p.d.f.
                self.subgraph.calculate_pdf(k, distance_function, self.pre_computed_distance, self.pre_distances)

                # Clustering with current `k` value
                self._clustering(k)

                # Performs the normalized cut with current `k` value
                cut = self._normalized_cut(k, distance_function)

                # If the cut's cost is smaller than minimum cut
                if cut < min_cut:
                    # Replace its value
                    min_cut = cut

                    # And defines a new best `k` value
                    best_k = k

        # Destroy current arcs
        self.subgraph.destroy_arcs()
        
        # Applying best `k` value to the subgraph
        self.subgraph.best_k = best_k

        # Creating new arcs with the best `k` value
        self.subgraph.create_arcs(best_k, distance_function, self.pre_computed_distance, self.pre_distances)

        # Calculating the new p.d.f. with the best `k` value
        self.subgraph.calculate_pdf(best_k, distance_function, self.pre_computed_distance, self.pre_distances)


    def fit(self, X):
        """Fits data in the classifier.

        Args:
            X (np.array): Array of features.

        """

        logger.info('Clustering with classifier ...')

        # Initializing the timer
        start = time.time()

        # Creating a subgraph
        self.subgraph = KNNSubgraph(X)

        # Checks if it is supposed to use pre-computed distances
        if self.pre_computed_distance:
            # Checks if its size is the same as the subgraph's amount of nodes
            if self.pre_distances.shape[0] != self.subgraph.n_nodes or self.pre_distances.shape[0] != self.subgraph.n_nodes:
                # If not, raises an error
                raise e.BuildError(
                    'Pre-computed distance matrix should have the size of `n_nodes x n_nodes`')

        # Performing the best minimum cut on the subgraph
        self._best_minimum_cut(self.min_k, self.max_k)

        # Clustering the data with best `k` value
        self._clustering(self.subgraph.best_k)

        # The subgraph has been properly trained
        self.subgraph.trained = True

        # Ending timer
        end = time.time()

        # Calculating training task time
        train_time = end - start

        logger.info('Classifier has been clustered with.')
        logger.info(f'Number of clusters: {self.subgraph.n_clusters}.')
        logger.info(f'Clustering time: {train_time} seconds.')
