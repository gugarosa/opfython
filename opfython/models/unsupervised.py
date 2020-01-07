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

    def _create_arcs(self, k_max):
        """
        """

        d = np.zeros(k_max+1)
        nn = np.zeros(k_max+1)
        maxdists = np.zeros(k_max)

        for i in range(self.subgraph.n_nodes):
            for l in range(k_max):
                d[l] = c.FLOAT_MAX
            for j in range(self.subgraph.n_nodes):
                if j != i:
                    if self.pre_computed_distance:
                        d[k_max] = self.pre_distances[self.subgraph.nodes[i].idx][self.subgraph.nodes[j].idx]
                    else:
                        d[k_max] = distance.DISTANCES[self.distance](self.subgraph.nodes[i].features, self.subgraph.nodes[j].features)
                    nn[k_max] = j
                    k = k_max
                    while k > 0 and d[k] < d[k-1]:
                        dist = d[k]
                        l = nn[k]
                        d[k] = d[k-1]
                        nn[k] = nn[k-1]
                        d[k-1] = dist
                        nn[k-1] = l
                        k -= 1
            self.subgraph.nodes[i].radius = 0.0
            self.subgraph.nodes[i].n_adjacency = 0
            for l in range(k_max-1, -1, -1):
                if d[l] != c.FLOAT_MAX:
                    if d[l] > self.subgraph.density:
                        self.subgraph.density = d[l]
                    if d[l] > self.subgraph.nodes[i].radius:
                        self.subgraph.nodes[i].radius = d[l]
                    if d[l] > maxdists[l]:
                        maxdists[l] = d[l]
                    self.subgraph.nodes[i].adjacency.append(nn[l])

        if self.subgraph.density < 0.00001:
            self.subgraph.density = 1

        return maxdists

    def _pdf_kmax(self):
        k_max = self.subgraph.best_k
        self.subgraph.K = 2 * self.subgraph.density / 9

        self.subgraph.min_density = c.FLOAT_MAX
        self.subgraph.max_density = c.FLOAT_MAX * -1


        value = np.zeros(self.subgraph.n_nodes)

        for i in range(self.subgraph.n_nodes):
            adjacency = self.subgraph.nodes[i].adjacency
            value[i] = 0
            n_elems = 1
            for k in range(k_max):
                if self.pre_computed_distance:
                    dist = self.pre_distances[self.subgraph.nodes[i].idx][self.subgraph.nodes[k].idx]
                else:
                    dist = distance.DISTANCES[self.distance](self.subgraph.nodes[i].features, self.subgraph.nodes[j].features)


    def _best_minimum_cut(self, k_min, k_max):
        maxdists = self._create_arcs(k_max)
        print(maxdists)

        # for k in range(k_min, k_max+1):
        #     if minimum_cut != 0:
        #         self.subgraph.density = maxdists[k-1]
        #         self.subgraph.best_k = k



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

        self._best_minimum_cut(1, 1)

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
