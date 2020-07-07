import copy
import time

import numpy as np

import opfython.math.general as g
import opfython.math.random as r
import opfython.utils.constants as c
import opfython.utils.exception as e
import opfython.utils.logging as l
from opfython.core import OPF, Heap, Subgraph

logger = l.get_logger(__name__)


class SupervisedOPF(OPF):
    """A SupervisedOPF which implements the supervised version of OPF classifier.

    References:
        J. P. Papa, A. X. Falcão and C. T. N. Suzuki. Supervised Pattern Classification based on Optimum-Path Forest.
        International Journal of Imaging Systems and Technology (2009).

    """

    def __init__(self, distance='log_squared_euclidean', pre_computed_distance=None):
        """Initialization method.

        Args:
            distance (str): An indicator of the distance metric to be used.
            pre_computed_distance (str): A pre-computed distance file for feeding into OPF.

        """

        logger.info('Overriding class: OPF -> SupervisedOPF.')

        # Override its parent class with the receiving arguments
        super(SupervisedOPF, self).__init__(distance, pre_computed_distance)

        logger.info('Class overrided.')

    def _find_prototypes(self):
        """Find prototype nodes using the Minimum Spanning Tree (MST) approach.

        """

        logger.debug('Finding prototypes ...')

        # Creating a Heap of size equals to number of nodes
        h = Heap(self.subgraph.n_nodes)

        # Marking first node without any predecessor
        self.subgraph.nodes[0].pred = c.NIL

        # Adding first node to the heap
        h.insert(0)

        # Creating a list of prototype nodes
        prototypes = []

        # While the heap is not empty
        while not h.is_empty():
            # Remove a node from the heap
            p = h.remove()

            # Gathers its cost from the heap
            self.subgraph.nodes[p].cost = h.cost[p]

            # And also its predecessor
            pred = self.subgraph.nodes[p].pred

            # If the predecessor is not NIL
            if pred != c.NIL:
                # Checks if the label of current node is the same as its predecessor
                if self.subgraph.nodes[p].label != self.subgraph.nodes[pred].label:
                    # If current node is not a prototype
                    if self.subgraph.nodes[p].status != c.PROTOTYPE:
                        # Marks it as a prototype
                        self.subgraph.nodes[p].status = c.PROTOTYPE

                        # Appends current node identifier to the prototype's list
                        prototypes.append(p)

                    # If predecessor node is not a prototype
                    if self.subgraph.nodes[pred].status != c.PROTOTYPE:
                        # Marks it as a protoype
                        self.subgraph.nodes[pred].status = c.PROTOTYPE

                        # Appends predecessor node identifier to the prototype's list
                        prototypes.append(pred)

            # For every possible node
            for q in range(self.subgraph.n_nodes):
                # Checks if the color of current node in the heap is not black
                if h.color[q] != c.BLACK:
                    # If `p` and `q` identifiers are different
                    if p != q:
                        # If it is supposed to use pre-computed distances
                        if self.pre_computed_distance:
                            # Gathers the arc from the distances' matrix
                            weight = self.pre_distances[self.subgraph.nodes[p].idx][self.subgraph.nodes[q].idx]

                        # If distance is supposed to be calculated
                        else:
                            # Calculates the distance
                            weight = self.distance_fn(
                                self.subgraph.nodes[p].features, self.subgraph.nodes[q].features)

                        # If current arc's cost is smaller than the path's cost
                        if weight < h.cost[q]:
                            # Marks `q` predecessor node as `p`
                            self.subgraph.nodes[q].pred = p

                            # Updates the arc on the heap
                            h.update(q, weight)

        logger.debug(f'Prototypes: {prototypes}.')

    def fit(self, X_train, Y_train, I_train=None):
        """Fits data in the classifier.

        Args:
            X_train (np.array): Array of training features.
            Y_train (np.array): Array of training labels.
            I_train (np.array): Array of training indexes.

        """

        logger.info('Fitting classifier ...')

        # Initializing the timer
        start = time.time()

        # Creating a subgraph
        self.subgraph = Subgraph(X_train, Y_train, I=I_train)

        # # Checks if it is supposed to use pre-computed distances
        # if self.pre_computed_distance:
        #     # Checks if its size is the same as the subgraph's amount of nodes
        #     if self.pre_distances.shape[0] != self.subgraph.n_nodes or self.pre_distances.shape[1] != self.subgraph.n_nodes:
        #         # If not, raises an error
        #         raise e.BuildError(
        #             'Pre-computed distance matrix should have the size of `n_nodes x n_nodes`')

        # Finding prototypes
        self._find_prototypes()

        # Creating a minimum heap
        h = Heap(size=self.subgraph.n_nodes)

        # For each possible node
        for i in range(self.subgraph.n_nodes):
            # Checks if node is a prototype
            if self.subgraph.nodes[i].status == c.PROTOTYPE:
                # If yes, it does not have predecessor nodes
                self.subgraph.nodes[i].pred = c.NIL

                # Its predicted label is the same as its true label
                self.subgraph.nodes[i].predicted_label = self.subgraph.nodes[i].label

                # Its cost equals to zero
                h.cost[i] = 0

                # Inserts the node into the heap
                h.insert(i)
            
            # If node is not a prototype
            else:
                # Its cost equals to maximum possible value
                h.cost[i] = c.FLOAT_MAX

        # While the heap is not empty
        while not h.is_empty():
            # Removes a node
            p = h.remove()

            # Appends its index to the ordered list
            self.subgraph.idx_nodes.append(p)

            # Gathers its cost
            self.subgraph.nodes[p].cost = h.cost[p]

            # For every possible node
            for q in range(self.subgraph.n_nodes):
                # If we are dealing with different nodes
                if p != q:
                    # If `p` node cost is smaller than `q` node cost
                    if h.cost[p] < h.cost[q]:
                        # Checks if we are using a pre-computed distance
                        if self.pre_computed_distance:
                            # Gathers the distance from the distance's matrix
                            weight = self.pre_distances[self.subgraph.nodes[p].idx][self.subgraph.nodes[q].idx]

                        # If the distance is supposed to be calculated
                        else:
                            # Calls the corresponding distance function
                            weight = self.distance_fn(
                                self.subgraph.nodes[p].features, self.subgraph.nodes[q].features)

                        # The current cost will be the maximum cost between the node's and its weight (arc)
                        current_cost = np.maximum(h.cost[p], weight)

                        # If current cost is smaller than `q` node's cost
                        if current_cost < h.cost[q]:
                            # `q` node has `p` as its predecessor
                            self.subgraph.nodes[q].pred = p

                            # And its predicted label is the same as `p`
                            self.subgraph.nodes[q].predicted_label = self.subgraph.nodes[p].predicted_label

                            # Updates the heap `q` node and the current cost
                            h.update(q, current_cost)

        # The subgraph has been properly trained
        self.subgraph.trained = True

        # Ending timer
        end = time.time()

        # Calculating training task time
        train_time = end - start

        logger.info('Classifier has been fitted.')
        logger.info(f'Training time: {train_time} seconds.')

    def predict(self, X_val, I_val=None):
        """Predicts new data using the pre-trained classifier.

        Args:
            X_val (np.array): Array of validation or test features.
            I_val (np.array): Array of validation or test indexes.

        Returns:
            A list of predictions for each record of the data.

        """

        # Checks if there is a subgraph
        if not self.subgraph:
            # If not, raises an BuildError
            raise e.BuildError('Subgraph has not been properly created')

        # Checks if subgraph has been properly trained
        if not self.subgraph.trained:
            # If not, raises an BuildError
            raise e.BuildError('Classifier has not been properly fitted')

        logger.info('Predicting data ...')

        # Initializing the timer
        start = time.time()

        # Creating a prediction subgraph
        pred_subgraph = Subgraph(X_val, I=I_val)

        # For every possible node
        for i in range(pred_subgraph.n_nodes):
            # Initializing the conqueror node
            conqueror = -1

            # Initializes the `j` counter
            j = 0

            # Gathers the first node from the ordered list
            k = self.subgraph.idx_nodes[j]

            # Checks if we are using a pre-computed distance
            if self.pre_computed_distance:
                # Gathers the distance from the distance's matrix
                weight = self.pre_distances[self.subgraph.nodes[k].idx][pred_subgraph.nodes[i].idx]

            # If the distance is supposed to be calculated
            else:
                # Calls the corresponding distance function
                weight = self.distance_fn(
                    self.subgraph.nodes[k].features, pred_subgraph.nodes[i].features)

            # The minimum cost will be the maximum between the `k` node cost and its weight (arc)
            min_cost = np.maximum(self.subgraph.nodes[k].cost, weight)

            # The current label will be `k` node's predicted label
            current_label = self.subgraph.nodes[k].predicted_label

            # While `j` is a possible node and the minimum cost is bigger than the current node's cost
            while j < (self.subgraph.n_nodes - 1) and min_cost > self.subgraph.nodes[self.subgraph.idx_nodes[j+1]].cost:
                # Gathers the next node from the ordered list
                l = self.subgraph.idx_nodes[j+1]

                # Checks if we are using a pre-computed distance
                if self.pre_computed_distance:
                    # Gathers the distance from the distance's matrix
                    weight = self.pre_distances[self.subgraph.nodes[l].idx][pred_subgraph.nodes[i].idx]

                # If the distance is supposed to be calculated
                else:
                    # Calls the corresponding distance function
                    weight = self.distance_fn(
                        self.subgraph.nodes[l].features, pred_subgraph.nodes[i].features)

                # The temporary minimum cost will be the maximum between the `l` node cost and its weight (arc)
                temp_min_cost = np.maximum(self.subgraph.nodes[l].cost, weight)

                # If temporary minimum cost is smaller than the minimum cost
                if temp_min_cost < min_cost:
                    # Replaces the minimum cost
                    min_cost = temp_min_cost

                    # Gathers the identifier of `l` node
                    conqueror = l

                    # Updates the current label as `l` node's predicted label
                    current_label = self.subgraph.nodes[l].predicted_label

                # Increments the `j` counter
                j += 1

                # Makes `k` and `l` equals
                k = l

            # Node's `i` predicted label is the same as current label
            pred_subgraph.nodes[i].predicted_label = current_label

            # Checks if any node has been conquered
            if conqueror > -1:
                # Marks the conqueror node and its path
                self.subgraph.mark_nodes(conqueror)

        # Creating the list of predictions
        preds = [pred.predicted_label for pred in pred_subgraph.nodes]

        # Ending timer
        end = time.time()

        # Calculating prediction task time
        predict_time = end - start

        logger.info('Data has been predicted.')
        logger.info(f'Prediction time: {predict_time} seconds.')

        return preds

    def learn(self, X_train, Y_train, X_val, Y_val, n_iterations=10):
        """Learns the best classifier over a validation set.

        Args:
            X_train (np.array): Array of training features.
            Y_train (np.array): Array of training labels.
            X_val (np.array): Array of validation features.
            Y_val (np.array): Array of validation labels.
            n_iterations (int): Number of iterations.

        """

        logger.info('Learning the best classifier ...')

        # Defines the maximum accuracy
        max_acc = 0

        # Defines the previous accuracy
        previous_acc = 0

        # Defines the iterations counter
        t = 0

        # An always true loop
        while True:
            logger.info(f'Running iteration {t+1}/{n_iterations} ...')

            # Fits training data into the classifier
            self.fit(X_train, Y_train)

            # Predicts new data
            preds = self.predict(X_val)

            # Calculating accuracy
            acc = g.opf_accuracy(Y_val, preds)

            # Checks if current accuracy is better than the best one
            if acc > max_acc:
                # If yes, replace the maximum accuracy
                max_acc = acc

                # Makes a copy of the best OPF classifier
                best_opf = copy.deepcopy(self)

                # And saves the iteration number
                best_t = t

            # Gathers which samples were missclassified
            errors = np.argwhere(Y_val != preds)

            # Defining the initial number of non-prototypes as 0
            non_prototypes = 0

            # For every possible subgraph's node
            for n in self.subgraph.nodes:
                # If the node is not a prototype
                if n.status != c.PROTOTYPE:
                    # Increments the number of non-prototypes
                    non_prototypes += 1

            # For every possible error
            for e in errors:
                # Counter will receive the number of non-prototypes
                ctr = non_prototypes

                # While the counter is bigger than zero
                while ctr > 0:
                    # Generates a random index
                    j = int(r.generate_uniform_random_number(0, len(X_train)))

                    # If the node on that particular index is not a prototype
                    if self.subgraph.nodes[j].status != c.PROTOTYPE:
                        # Swap the input nodes
                        X_train[j, :], X_val[e, :] = X_val[e, :], X_train[j, :]

                        # Swap the target nodes
                        Y_train[j], Y_val[e] = Y_val[e], Y_train[j]

                        # Decrements the number of non-prototypes
                        non_prototypes -= 1

                        # Resets the counter
                        ctr = 0

                    # If the node on that particular index is a prototype
                    else:
                        # Decrements the counter
                        ctr -= 1

            # Calculating difference between current accuracy and previous one
            delta = np.fabs(acc - previous_acc)

            # Replacing the previous accuracy as current accuracy
            previous_acc = acc

            # Incrementing the counter
            t += 1

            logger.info(
                f'Accuracy: {acc} | Delta: {delta} | Maximum Accuracy: {max_acc}')

            # If the difference is smaller than 10e-4 or iterations are finished
            if delta < 0.0001 or t == n_iterations:
                # Replaces current class with the best OPF
                self = best_opf

                logger.info(
                    f'Best classifier has been learned over iteration {best_t+1}.')

                # Breaks the loop
                break

    def prune(self, X_train, Y_train, X_val, Y_val, n_iterations=10):
        """Prunes a classifier over a validation set.

        Args:
            X_train (np.array): Array of training features.
            Y_train (np.array): Array of training labels.
            X_val (np.array): Array of validation features.
            Y_val (np.array): Array of validation labels.
            n_iterations (int): Maximum number of iterations.

        """

        logger.info('Pruning classifier ...')

        # Fits training data into the classifier
        self.fit(X_train, Y_train)

        # Predicts new data
        self.predict(X_val)

        # Gathering initial number of nodes
        initial_nodes = self.subgraph.n_nodes

        # For every possible iteration
        for t in range(n_iterations):
            logger.info(f'Running iteration {t+1}/{n_iterations} ...')

            # Creating temporary lists
            X_temp, Y_temp = [], []

            # Removing irrelevant nodes
            for j, n in enumerate(self.subgraph.nodes):
                if n.relevant != c.IRRELEVANT:
                    X_temp.append(X_train[j, :])
                    Y_temp.append(Y_train[j])

            # Copying lists back to original data
            X_train = np.asarray(X_temp)
            Y_train = np.asarray(Y_temp)

            # Fits training data into the classifier
            self.fit(X_train, Y_train)

            # Predicts new data
            preds = self.predict(X_val)

            # Calculating accuracy
            acc = g.opf_accuracy(Y_val, preds)

            logger.info(f'Current accuracy: {acc}.')

        # Gathering final number of nodes
        final_nodes = self.subgraph.n_nodes

        # Calculating pruning ratio
        prune_ratio = 1 - final_nodes / initial_nodes

        logger.info(f'Prune ratio: {prune_ratio}.')
