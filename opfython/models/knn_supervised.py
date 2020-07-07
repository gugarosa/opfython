import time

import numpy as np

import opfython.math.general as g
import opfython.utils.constants as c
import opfython.utils.exception as e
import opfython.utils.logging as l
from opfython.core import OPF, Heap
from opfython.subgraphs import KNNSubgraph

logger = l.get_logger(__name__)


class KNNSupervisedOPF(OPF):
    """A KNNSupervisedOPF which implements the supervised version of OPF classifier with a KNN subgraph.

    References:
        J. P. Papa and A. X. Falcão. A Learning Algorithm for the Optimum-Path Forest Classifier.
        Graph-Based Representations in Pattern Recognition (2009).

    """

    def __init__(self, max_k=1, distance='log_squared_euclidean', pre_computed_distance=None):
        """Initialization method.

        Args:
            max_k (int): Maximum `k` value for cutting the subgraph.
            distance (str): An indicator of the distance metric to be used.
            pre_computed_distance (str): A pre-computed distance file for feeding into OPF.

        """

        logger.info('Overriding class: OPF -> KNNSupervisedOPF.')

        # Override its parent class with the receiving arguments
        super(KNNSupervisedOPF, self).__init__(distance, pre_computed_distance)

        # Defining the maximum `k` value for cutting the subgraph
        self.max_k = max_k

        logger.info('Class overrided.')

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

    def _clustering(self, force_prototype=False):
        """Clusters the subgraph.

        Args:
            force_prototype (bool): Whether clustering should for each class to have at least one prototype.

        """

        # For every possible node
        for i in range(self.subgraph.n_nodes):
            # For every adjacent node of `i`
            for j in self.subgraph.nodes[i].adjacency:
                # Making sure that variable is an integer
                j = int(j)

                # Checks if node `i` density is equals as node `j` density
                if self.subgraph.nodes[i].density == self.subgraph.nodes[j].density:
                    # Marks the insertion flag as True
                    insert = True

                    # For every adjacent node of `j`
                    for l in self.subgraph.nodes[j].adjacency:
                        # Making sure that variable is an integer
                        l = int(l)

                        # Checks if it is the same node as `i`
                        if i == l:
                            # If yes, mark insertion flag as False
                            insert = False

                    # If insertion flag is True
                    if insert:
                        # Inserts node `i` in the adjacency list of `j`
                        self.subgraph.nodes[j].adjacency.insert(0, i)

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

                # Defines its predicted label as the node's true label
                self.subgraph.nodes[p].predicted_label = self.subgraph.nodes[p].label

            # Apply current node's cost as the heap's cost
            self.subgraph.nodes[p].cost = h.cost[p]

            # For every possible adjacent node
            for q in self.subgraph.nodes[p].adjacency:
                # Making sure that variable is an integer
                q = int(q)

                # If its color in the heap is different from `BLACK`
                if h.color[q] != c.BLACK:
                    # Calculates the current cost
                    current_cost = np.minimum(
                        h.cost[p], self.subgraph.nodes[q].density)

                    # If prototypes should be forced to belong to a class
                    if force_prototype:
                        # Checks if nodes `p` and `q` labels are different
                        if self.subgraph.nodes[p].label != self.subgraph.nodes[q].label:
                            # If yes, define current cost as minimum value possible
                            current_cost = -c.FLOAT_MAX

                    # If current cost is bigger than heap's cost
                    if current_cost > h.cost[q]:
                        # Apply `q` predecessor as `p`
                        self.subgraph.nodes[q].pred = p

                        # Gathers the same root's identifier
                        self.subgraph.nodes[q].root = self.subgraph.nodes[p].root

                        # And its cluster label
                        self.subgraph.nodes[q].predicted_label = self.subgraph.nodes[p].predicted_label

                        # Updates node `q` on the heap with the current cost
                        h.update(q, current_cost)

    def _learn(self, X_train, Y_train, I_train, X_val, Y_val, I_val):
        """Learns the best `k` value over the validation set.

        Args:
            X_train (np.array): Array of training features.
            Y_train (np.array): Array of training labels.
            I_train (np.array): Array of training indexes.
            X_val (np.array): Array of validation features.
            Y_val (np.array): Array of validation labels.
            I_val (np.array): Array of validation indexes.

        """

        logger.info('Learning best `k` value ...')

        # Creating a subgraph
        self.subgraph = KNNSubgraph(X_train, Y_train, I_train)

        # Checks if it is supposed to use pre-computed distances
        if self.pre_computed_distance:
            # Checks if its size is the same as the subgraph's amount of nodes
            if self.pre_distances.shape[0] != self.subgraph.n_nodes or self.pre_distances.shape[1] != self.subgraph.n_nodes:
                # If not, raises an error
                raise e.BuildError(
                    'Pre-computed distance matrix should have the size of `n_nodes x n_nodes`')

        # Defining initial maximum accuracy as 0
        max_acc = 0.0

        # For every possible `k` value
        for k in range(1, self.max_k + 1):
            # Gathers current `k` as subgraph's best `k`
            self.subgraph.best_k = k

            # Calculate the arcs using the current `k` value
            self.subgraph.create_arcs(
                k, self.distance_fn, self.pre_computed_distance, self.pre_distances)

            # Calculate the p.d.f. using the current `k` value
            self.subgraph.calculate_pdf(
                k, self.distance_fn, self.pre_computed_distance, self.pre_distances)

            # Clusters the subgraph
            self._clustering()

            # Calculate the predictions over the validation set
            preds = self.predict(X_val, I_val)

            # Calculating the accuracy
            acc = g.opf_accuracy(Y_val, preds)

            # If accuracy is better than maximum accuracy
            if acc > max_acc:
                # Replaces the maximum accuracy value
                max_acc = acc

                # Defines current `k` as the best `k` value
                best_k = k

            logger.info(f'Accuracy over k = {k}: {acc}')

            # Destroy the arcs
            self.subgraph.destroy_arcs()

        # Applying the best k to the subgraph's property
        self.subgraph.best_k = best_k

    def fit(self, X_train, Y_train, X_val, Y_val, I_train=None, I_val=None):
        """Fits data in the classifier.

        Args:
            X_train (np.array): Array of training features.
            Y_train (np.array): Array of training labels.
            X_val (np.array): Array of validation features.
            Y_val (np.array): Array of validation labels.
            I_train (np.array): Array of training indexes.
            I_val (np.array): Array of validation indexes.

        """

        logger.info('Fitting classifier ...')

        # Initializing the timer
        start = time.time()

        # Performing the learning process in order to find the best `k` value
        self._learn(X_train, Y_train, I_train, X_val, Y_val, I_val)

        # Creating arcs with the best `k` value
        self.subgraph.create_arcs(
            self.subgraph.best_k, self.distance_fn, self.pre_computed_distance, self.pre_distances)

        # Calculating p.d.f. with the best `k` value
        self.subgraph.calculate_pdf(
            self.subgraph.best_k, self.distance_fn, self.pre_computed_distance, self.pre_distances)

        # Clustering subgraph forcing each class to have at least one prototype
        self._clustering(force_prototype=True)

        # Destroying arcs
        self.subgraph.destroy_arcs()

        # The subgraph has been properly trained
        self.subgraph.trained = True

        # Ending timer
        end = time.time()

        # Calculating training task time
        train_time = end - start

        logger.info(f'Classifier has been fitted with k = {self.subgraph.best_k}.')
        logger.info(f'Training time: {train_time} seconds.')

    def predict(self, X_test, I_test=None, verbose=False):
        """Predicts new data using the pre-trained classifier.

        Args:
            X_test (np.array): Array of features.
            I_test (np.array): Array of indexes.

        Returns:
            A list of predictions for each record of the data.

        """

        logger.info('Predicting data ...')

        # Initializing the timer
        start = time.time()

        # Creating a prediction subgraph
        pred_subgraph = KNNSubgraph(X_test, I=I_test)

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
                        distances[best_k] = self.distance_fn(
                            pred_subgraph.nodes[i].features, self.subgraph.nodes[j].features)

                    # Apply node `j` as a neighbour
                    neighbours_idx[best_k] = j

                    # Gathers current `k`
                    cur_k = best_k

                    # While current `k` is bigger than 0 and the `k` distance is smaller than `k-1` distance
                    while cur_k > 0 and distances[cur_k] < distances[cur_k - 1]:
                        # Swaps the distance from `k` and `k-1`
                        distances[cur_k], distances[cur_k - 1] = distances[cur_k - 1], distances[cur_k]

                        # Swaps the neighbours indexex from `k` and `k-1`
                        neighbours_idx[cur_k], neighbours_idx[cur_k - 1] = neighbours_idx[cur_k - 1], neighbours_idx[cur_k]

                        # Decrements `k`
                        cur_k -= 1

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
                       (self.subgraph.max_density - self.subgraph.min_density + c.EPSILON)) + 1

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
