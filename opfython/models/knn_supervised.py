import time

import numpy as np

import opfython.math.distance as distance
import opfython.math.general as g
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

    def _clustering(self, force_prototype=False):
        """Clusters the subgraph using the best `k` value.

        Args:
            best_k (int): Best value of k.

        """

        # For every possible node
        for i in range(self.subgraph.n_nodes):
            for i_adj in self.subgraph.nodes[i].adjacency:
                 if self.subgraph.nodes[i].density == self.subgraph.nodes[int(i_adj)].density:
                     insert = True
                     for j_adj in self.subgraph.nodes[int(i_adj)].adjacency:
                        if i == j_adj:
                            insert = False
                        if insert:
                            # Inserts node `i` in the adjacency list of `i_adj`
                            self.subgraph.nodes[i_adj].adjacency.insert(0, i)


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

                # print(h.cost[p])

                # Defines its cluster label as `l`
                self.subgraph.nodes[p].predicted_label = self.subgraph.nodes[p].label

            # Apply current node's cost as the heap's cost
            self.subgraph.nodes[p].cost = h.cost[p]

            for p_adj in self.subgraph.nodes[p].adjacency:
                # If its color in the heap is different from `BLACK`
                if h.color[int(p_adj)] != c.BLACK:
                    # Calculates the current cost
                    current_cost = np.minimum(
                        h.cost[p], self.subgraph.nodes[int(p_adj)].density)

                    if force_prototype:
                        if self.subgraph.nodes[p].label != self.subgraph.nodes[int(p_adj)].label:
                            current_cost = c.FLOAT_MAX * -1

                    # If temporary cost is bigger than heap's cost
                    if current_cost > h.cost[int(p_adj)]:
                        # Apply `int(p_adj)` predecessor as `p`
                        self.subgraph.nodes[int(p_adj)].pred = p

                        # Gathers the same root's identifier
                        self.subgraph.nodes[int(p_adj)].root = self.subgraph.nodes[p].root

                        # And its cluster label
                        self.subgraph.nodes[int(p_adj)].predicted_label = self.subgraph.nodes[p].predicted_label

                        #
                        h.update(int(p_adj), current_cost)

    def _learn(self, X_train, Y_train, X_val, Y_val):

        # Gathers the distance function
        distance_function = distance.DISTANCES[self.distance]

        # Creating a subgraph
        self.subgraph = KNNSubgraph(X_train, Y_train)

        max_acc = 0

        for k in range(1, self.max_k + 1):
            self.subgraph.best_k = k
            self.subgraph.create_arcs(k, distance_function, self.pre_computed_distance, self.pre_distances)

            self.subgraph.calculate_pdf(k, distance_function, self.pre_computed_distance, self.pre_distances)

            self._clustering()

            preds = self.predict(X_val)

            acc = g.opf_accuracy(Y_val, preds)

            print(acc, k)

            if acc > max_acc:
                max_acc = acc
                best_k = k
                self.subgraph.best_k = k

            self.subgraph.destroy_arcs()

        return best_k

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
        # self.subgraph = KNNSubgraph(X_train, Y_train)

        # Checks if it is supposed to use pre-computed distances
        if self.pre_computed_distance:
            # Checks if its size is the same as the subgraph's amount of nodes
            if self.pre_distances.shape[0] != self.subgraph.n_nodes or self.pre_distances.shape[0] != self.subgraph.n_nodes:
                # If not, raises an error
                raise e.BuildError(
                    'Pre-computed distance matrix should have the size of `n_nodes x n_nodes`')

        #
        self.subgraph.best_k = self._learn(X_train, Y_train, X_val, Y_val)

        # Gathers the distance function
        distance_function = distance.DISTANCES[self.distance]

        self.subgraph.create_arcs(self.subgraph.best_k, distance_function, self.pre_computed_distance, self.pre_distances)

        self.subgraph.calculate_pdf(self.subgraph.best_k, distance_function, self.pre_computed_distance, self.pre_distances)

        self._clustering(force_prototype=True)

        self.subgraph.destroy_arcs()



        # The subgraph has been properly trained
        self.subgraph.trained = True

        # Ending timer
        end = time.time()

        # Calculating training task time
        train_time = end - start

        logger.info('Classifier has been fitted.')
        logger.info(f'Training time: {train_time} seconds.')

    def predict(self, X):
        """Predicts new data using the pre-trained classifier.

        Args:
            X (np.array): Array of features.

        Returns:
            A list of predictions for each record of the data.

        """



        logger.info('Predicting data ...')

        # Initializing the timer
        start = time.time()

        # Creating a prediction subgraph
        pred_subgraph = KNNSubgraph(X)

        # Gathering the best `k` value
        best_k = self.subgraph.best_k

        # Creating an array of distances
        distances = np.zeros(best_k + 1)

        # Creating an array of nearest neighbours indexes
        neighbours_idx = np.zeros(best_k + 1)

        # For every possible prediction node
        for i in range(pred_subgraph.n_nodes):
            # Defines the current cost
            cost = c.FLOAT_MAX * -1

            # Filling array of distances with maximum value
            distances.fill(c.FLOAT_MAX)

            # For every possible trained node
            for j in range(self.subgraph.n_nodes):
                # If they are different nodes
                if j != i:
                    # If it is supposed to use a pre-computed distance
                    if self.pre_computed_distance:
                        # Gathers the distance from the matrix
                        distances[best_k] = self.pre_distances[pred_subgraph.nodes[i].idx][self.subgraph.nodes[j].idx]

                    # If it is supposed to calculate the distance
                    else:
                        # Calculates the distance between nodes `i` and `j`
                        distances[best_k] = distance.DISTANCES[self.distance](
                            pred_subgraph.nodes[i].features, self.subgraph.nodes[j].features)

                    # Apply node `j` as a neighbour
                    neighbours_idx[best_k] = j

                    # Gathers current `k`
                    current_k = best_k

                    # While current `k` is bigger than 0 and the `k` distance is smaller than `k-1` distance
                    while current_k > 0 and distances[current_k] < distances[current_k - 1]:
                        # Swaps the distance from `k` and `k-1`
                        distances[current_k], distances[current_k -
                                                        1] = distances[current_k - 1], distances[current_k]

                        # Swaps the neighbours indexex from `k` and `k-1`
                        neighbours_idx[current_k], neighbours_idx[current_k -
                                                                  1] = neighbours_idx[current_k - 1], neighbours_idx[current_k]

                        # Decrements `k`
                        current_k -= 1

            # Defining the density as 0
            density = 0.0

            # For every possible k
            for k in range(best_k):
                # Accumulates the density
                density += np.exp(-distances[k] / self.subgraph.constant)

            # Gather its mean value
            density /= best_k

            # Scale the density between minimum and maximum values
            density = ((c.MAX_DENSITY - 1) * (density - self.subgraph.min_density) /
                       (self.subgraph.max_density - self.subgraph.min_density)) + 1

            # For every possible k
            for k in range(best_k):
                # If distance is different than maximum possible value
                if distances[k] != c.FLOAT_MAX:
                    # Gathers the node's neighbour
                    neighbour = int(neighbours_idx[k])

                    # Calculate the temporary cost
                    temp_cost = np.minimum(
                        self.subgraph.nodes[neighbour].cost, density)

                    # If temporary cost is bigger than current cost
                    if temp_cost > cost:
                        # Replaces the current cost
                        cost = temp_cost

                        # And propagates the predicted label from the neighbour
                        pred_subgraph.nodes[i].predicted_label = self.subgraph.nodes[neighbour].predicted_label

        # Creating the list of predictions
        preds = [pred.predicted_label for pred in pred_subgraph.nodes]

        # Ending timer
        end = time.time()

        # Calculating prediction task time
        predict_time = end - start

        logger.info('Data has been predicted.')
        logger.info(f'Prediction time: {predict_time} seconds.')

        return preds