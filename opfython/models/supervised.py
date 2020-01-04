import time
import numpy as np

import opfython.math.distance as d
import opfython.utils.constants as c
import opfython.utils.exception as e
import opfython.utils.logging as l
from opfython.core.heap import Heap
from opfython.core.opf import OPF
from opfython.core.subgraph import Subgraph

logger = l.get_logger(__name__)


class SupervisedOPF(OPF):
    """A SupervisedOPF which implements the supervised version of OPF classifier.

    References:
        J. P. Papa, A. X. Falcão and C. T. N. Suzuki. Supervised Pattern Classification based on Optimum-Path Forest. International Journal of Imaging Systems and Technology (2009).

    """

    def __init__(self, pre_computed_distance=False):
        """Initialization method.

        Args:
            pre_computed_distance (bool): Whether OPF should use pre-computed distances or not.

        """

        logger.info('Overriding class: OPF -> SupervisedOPF.')

        # Override its parent class with the receiving arguments
        super(SupervisedOPF, self).__init__(
            pre_computed_distance=pre_computed_distance)

        logger.info('Class overrided.')

    def fit(self, X, Y):
        """Fits data in the classifier.

        Args:
            X (np.array): Array of features.
            Y (np.array): Array of labels.

        """

        logger.info('Fitting classifier ...')

        # Initializing the timer
        start = time.time()

        # Creating a subgraph
        self.g = Subgraph(X, Y)

        # Finding prototypes
        self._find_prototypes(self.g)

        # Creating a minimum heap
        h = Heap(size=self.g.n_nodes)

        # Also creating an costs array
        costs = np.zeros(self.g.n_nodes)

        # Filling the costs array with maximum possible value
        costs.fill(c.FLOAT_MAX)

        # For each possible node
        for i in range(self.g.n_nodes):
            # Checks if node is a prototype or not
            if self.g.nodes[i].status == c.PROTOTYPE:
                print(i)
                # If yes, it does not have predecessor nodes
                self.g.nodes[i].pred = c.NIL

                # Its predicted label is the same as its true label
                self.g.nodes[i].predicted_label = self.g.nodes[i].label

                # Its cost equals to zero
                costs[i] = 0

                # Inserts the node into the heap
                h.insert(i)

        # Resets the `i` counter
        i = 0

        # While the heap is not empty
        while not h.is_empty():
            # Removes a node
            p = h.remove()

            # Appends its index to the ordered list
            self.g.idx_nodes.append(p)

            # Gathers its cost
            self.g.nodes[p].cost = costs[p]

            # Increases the counter
            i += 1

            # For every possible node
            for q in range(self.g.n_nodes):
                # If we are dealing with different nodes
                if p is not q:
                    # If `p` node cost is smaller than `q` node cost
                    if costs[p] < costs[q]:
                        # Checks if we are using a pre-computed distance
                        if self.pre_computed_distance:
                            # Gathers the distance from the distance's matrix
                            weight = self.distances[self.g.nodes[p].idx][self.g.nodes[q].idx]

                        # If the distance is supposed to be calculated
                        else:
                            # Calls the corresponding distance function
                            weight = d.log_squared_euclidean_distance(
                                self.g.nodes[p].features, self.g.nodes[q].features)

                        # The current cost will be the maximum cost between the node's and its weight (arc)
                        current_cost = np.maximum(costs[p], weight)

                        # If current cost is smaller than `q` node's cost
                        if current_cost < costs[q]:
                            # `q` node has `p` as its predecessor
                            self.g.nodes[q].pred = p

                            # And its predicted label is the same as `p`
                            self.g.nodes[q].predicted_label = self.g.nodes[p].predicted_label

                            # Updates the heap with `q` node and the current cost
                            h.update(q, current_cost)

        # The subgraph has been properly trained
        self.g.trained = True

        # Ending timer
        end = time.time()

        # Calculating training task time
        train_time = end - start

        logger.info('Classifier has been trained.')
        logger.info(f'Training time: {train_time} seconds.')

    def predict(self, X):
        """Predicts new data using the pre-trained classifier.

        Args:
            X (np.array): Array of features.

        Returns:
            A list of predictions for each record of the data.

        """

        # Checks if there is a subgraph
        if not self.g:
            # If not, raises an BuildError
            raise e.BuildError('Subgraph has not been properly created')

        # Checks if subgraph has been properly trained
        if not self.g.trained:
            # If not, raises an BuildError
            raise e.BuildError('Classifier has not been properly trained')

        logger.info('Predicting data ...')

        # Initializing the timer
        start = time.time()

        # Creating a prediction subgraph
        g_predict = Subgraph(X)

        # For every possible node
        for i in range(g_predict.n_nodes):
            # Initializes the `j` counter
            j = 0

            # Gathers the first node from the ordered list
            k = self.g.idx_nodes[j]

            # Checks if we are using a pre-computed distance
            if self.pre_computed_distance:
                # Gathers the distance from the distance's matrix
                weight = self.distances[self.g.nodes[k].idx][g_predict.nodes[i].idx]

            # If the distance is supposed to be calculated
            else:
                # Calls the corresponding distance function
                weight = d.log_squared_euclidean_distance(
                    self.g.nodes[k].features, g_predict.nodes[i].features)

            # The minimum cost will be the maximum between the `k` node cost and its weight (arc)
            min_cost = np.maximum(self.g.nodes[k].cost, weight)

            # The current label will be `k` node's predicted label
            current_label = self.g.nodes[k].predicted_label

            # While `j` is a possible node and the minimum cost is bigger than the current node's cost
            while j < (self.g.n_nodes - 1) and min_cost > self.g.nodes[self.g.idx_nodes[j+1]].cost:
                # Gathers the next node from the ordered list
                l = self.g.idx_nodes[j+1]

                # Checks if we are using a pre-computed distance
                if self.pre_computed_distance:
                    # Gathers the distance from the distance's matrix
                    weight = self.distances[self.g.nodes[l].idx][g_predict.nodes[i].idx]

                # If the distance is supposed to be calculated
                else:
                    # Calls the corresponding distance function
                    weight = d.log_squared_euclidean_distance(
                        self.g.nodes[l].features, g_predict.nodes[i].features)

                # The temporary minimum cost will be the maximum between the `l` node cost and its weight (arc)
                temp_min_cost = np.maximum(self.g.nodes[l].cost, weight)

                # If temporary minimum cost is smaller than the minimum cost
                if temp_min_cost < min_cost:
                    # Replaces the minimum cost
                    min_cost = temp_min_cost

                    # Updates the current label as `l` node's predicted label
                    current_label = self.g.nodes[l].predicted_label

                # Increments the `j` counter
                j += 1

                # Makes `k` and `l` equals
                k = l

            # Node's `i` predicted label is the same as current label
            g_predict.nodes[i].predicted_label = current_label

        # Creating the list of predictions
        preds = [pred.predicted_label for pred in g_predict.nodes]

        # Ending timer
        end = time.time()

        # Calculating prediction task time
        predict_time = end - start

        logger.info('Data has been predicted.')
        logger.info(f'Prediction time: {predict_time} seconds.')

        return preds
