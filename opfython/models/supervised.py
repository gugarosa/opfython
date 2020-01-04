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

        #
        start = time.time()

        # Creating a subgraph
        self.g = Subgraph(X, Y)

        # Finding prototypes
        self._find_prototypes(self.g)

        #
        h = Heap(size=self.g.n_nodes)

        #
        costs = np.zeros(self.g.n_nodes)

        #
        costs.fill(c.FLOAT_MAX)

        #
        for i in range(self.g.n_nodes):
            if self.g.nodes[i].status == c.PROTOTYPE:
                self.g.nodes[i].pred = c.NIL
                costs[i] = 0
                self.g.nodes[i].predicted_label = self.g.nodes[i].label
                h.insert(i)

        i = 0

        while not h.is_empty():
            p = h.remove()
            self.g.idx_nodes.append(p)
            i += 1
            self.g.nodes[p].cost = costs[p]

            for q in range(self.g.n_nodes):
                if not p == q:
                    if costs[p] < costs[q]:
                        if self.pre_computed_distance:
                            weight = self.distances[self.g.nodes[p].idx][self.g.nodes[q].idx]
                        else:
                            weight = d.log_squared_euclidean_distance(
                                self.g.nodes[p].features, self.g.nodes[q].features)
                        
                        current_cost = np.maximum(costs[p], weight)
                        if current_cost < costs[q]:
                            self.g.nodes[q].pred = p
                            self.g.nodes[q].predicted_label = self.g.nodes[p].predicted_label
                            h.update(q, current_cost)

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

        if not self.g:
            raise e.BuildError('Subgraph has not been properly created')
        
        if not self.g.trained:
            raise e.BuildError('Classifier has not been properly trained')

        logger.info('Predicting data ...')

        start = time.time()

        for i in range(self.g.n_nodes):
            j = 0
            k = self.g.idx_nodes[j]
            if self.pre_computed_distance:
                weight = self.distances[self.g.nodes[k].idx][self.g.nodes[i].idx]
            else:
                weight = d.log_squared_euclidean_distance(
                    self.g.nodes[k].features, self.g.nodes[i].features)
            
            min_cost = np.maximum(self.g.nodes[k].cost, weight)

            current_label = self.g.nodes[k].predicted_label

            while (j < self.g.n_nodes - 1) & (min_cost > self.g.nodes[self.g.idx_nodes[j+1]].cost):
                l = self.g.idx_nodes[j+1]

                if self.pre_computed_distance:
                    weight = self.distances[self.g.nodes[l].idx][self.g.nodes[i].idx]
                else:
                    weight = d.log_squared_euclidean_distance(
                        self.g.nodes[l].features, self.g.nodes[i].features)

                temp_min_cost = np.maximum(self.g.nodes[l].cost, weight)

                if (temp_min_cost < min_cost):
                    min_cost = temp_min_cost
                    current_label = self.g.nodes[l].predicted_label
                j += 1
                k = l
            
            self.g.nodes[i].predicted_label = current_label

        # Ending timer
        end = time.time()

        # Calculating training task time
        predict_time = end - start

        logger.info('Data has been predicted.')
        logger.info(f'Prediction time: {predict_time} seconds.')
