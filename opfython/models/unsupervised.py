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

    def __init__(self, distance='log_squared_euclidean', pre_computed_distance=None):
        """Initialization method.

        Args:
            distance (str): An indicator of the distance metric to be used.
            pre_computed_distance (str): A pre-computed distance file for feeding into OPF.

        """

        logger.info('Overriding class: OPF -> UnsupervisedOPF.')

        # Override its parent class with the receiving arguments
        super(UnsupervisedOPF, self).__init__(
            distance=distance, pre_computed_distance=pre_computed_distance)

        logger.info('Class overrided.')

    def _clustering(self, k):
        """
        """

        for i in range(self.subgraph.n_nodes):
            i_adjacency = self.subgraph.nodes[i].adjacency
            for _ in range(k):
                for i_adj in i_adjacency:
                    if self.subgraph.nodes[i].density == self.subgraph.nodes[int(i_adj)].density:
                        j_adjacency = self.subgraph.nodes[int(i_adj)].adjacency
                        insert = True
                        for _ in range(k):
                            for j_adj in j_adjacency:
                                if i == j_adj:
                                    insert = False
                        if insert:
                            self.subgraph.nodes[int(i_adj)].adjacency.append(i)
                            self.subgraph.nodes[int(i_adj)].n_adjacency += 1

        # Creating a minimum heap
        h = Heap(size=self.subgraph.n_nodes, policy='max')

        # Also creating an costs array
        costs = np.zeros(self.subgraph.n_nodes)

        for i in range(self.subgraph.n_nodes):
            costs[i] = self.subgraph.nodes[i].cost
            self.subgraph.nodes[i].pred = c.NIL
            self.subgraph.nodes[i].root = i
            h.insert(i)

        # Resets the `i` counter
        i = 0
        l = 0

        while not h.is_empty():
            # Removes a node
            p = h.remove()

            # Appends its index to the ordered list
            self.subgraph.idx_nodes.append(p)

            if self.subgraph.nodes[p].pred == c.NIL:
                costs[p] = self.subgraph.nodes[p].density
                self.subgraph.nodes[p].predicted_label = l
                l += 1

            self.subgraph.nodes[p].cost = costs[p]

            n_adjacents = self.subgraph.nodes[p].n_adjacency + k

            for p_adj in self.subgraph.nodes[p].adjacency:
                if h.color[int(p_adj)] != c.BLACK:
                    temp = np.minimum(costs[p], self.subgraph.nodes[int(p_adj)].density)
                    if temp > costs[int(p_adj)]:
                        temp = costs[int(p_adj)]
                        self.subgraph.nodes[int(p_adj)].pred = p
                        self.subgraph.nodes[int(p_adj)].root = self.subgraph.nodes[p].root
                        self.subgraph.nodes[int(p_adj)].predicted_label = self.subgraph.nodes[p].predicted_label

        self.subgraph.n_labels = l

        # print(l)

            
        

    def _best_minimum_cut(self, k_min, k_max):
        """Performs a minimum cut on the subgraph using the best `k` value.

        Args:
            k_min (int): Minimum value of k.
            k_max (int): Maximum value of k.

        """

        # Gathers the distance function
        distance_function = distance.DISTANCES[self.distance]

        # Calculates the maximum possible distances
        max_distances = self.subgraph.create_arcs(k_max, distance_function, self.pre_computed_distance, self.pre_distances)

        # Initialize the minimum cut as maximum possible value
        min_cut = c.FLOAT_MAX
        
        # For every possible value of `k`
        for k in range(k_min, k_max + 1):
            # If minimum cut is different than zero
            if min_cut != 0:
                # Gathers the subgraph's density
                self.subgraph.density = max_distances[k - 1]

                # Gathers the subgraph's best `k` value
                self.subgraph.best_k = k

                # Calculates the p.d.f.
                self.subgraph.calculate_pdf(k, distance_function, self.pre_computed_distance, self.pre_distances)

                #
                self._clustering(k)


    def fit(self, X):
        """Fits data in the classifier.

        Args:
            X (np.array): Array of features.

        """

        logger.info('Fitting classifier ...')

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

        #
        self._best_minimum_cut(1, 2)

        #
        # for i in range(self.subgraph.n_nodes):
        #     i_adjacents = self.subgraph.nodes[i].adjacent
        #     while i_adjacents is not None:
        #         j = i_adjacents[0]
        #         if self.subgraph.nodes[i].density == self.subgraph.nodes[j].density:
        #             j_adjacentes = self.subgraph.nodes[j].adjacent
        #             insert_i = 1
        #             while j_adjacentes is not None:
        #                 if i == j_adjacentes[0]:

        # for i in range(self.subgraph.n_nodes):
        #     left_adjacency = self.subgraph.nodes[i].adjacency
        #     for left in left_adjacency:
        #         j = left
        #         if self.subgraph.nodes[i].density == self.subgraph.nodes[j].density:
        #             right_adjacency = self.subgraph.nodes[j].adjacency
        #             insert = True
        #             for right in right_adjacency:
        #                 if i == right:
        #                     insert = False
        #                     break
        #             if insert:
        #                 self.subgraph.nodes[j].adjacency.append(i)
                    



        # # The subgraph has been properly trained
        # self.subgraph.trained = True

        # # Ending timer
        # end = time.time()

        # # Calculating training task time
        # train_time = end - start

        # logger.info('Classifier has been fitted.')
        # logger.info(f'Training time: {train_time} seconds.')
