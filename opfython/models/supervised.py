"""Supervised Optimum-Path Forest.
"""

import copy
import time
from typing import List, Optional

import numpy as np

import opfython.math.general as g
import opfython.math.random as r
import opfython.utils.constants as c
import opfython.utils.exception as e
from opfython.core import OPF, Heap, Subgraph
from opfython.utils import logging

logger = logging.get_logger(__name__)


class SupervisedOPF(OPF):
    """A SupervisedOPF which implements the supervised version of OPF classifier.

    References:
        J. P. Papa, A. X. Falcão and C. T. N. Suzuki. Supervised Pattern Classification based on Optimum-Path Forest.
        International Journal of Imaging Systems and Technology (2009).

    """

    def __init__(
        self,
        distance: str = "log_squared_euclidean",
        pre_computed_distance: Optional[str] = None,
    ) -> None:
        """Initialization method.

        Args:
            distance: An indicator of the distance metric to be used.
            pre_computed_distance: A pre-computed distance file for feeding into OPF.

        """

        logger.info("Overriding class: OPF -> SupervisedOPF.")

        super(SupervisedOPF, self).__init__(distance, pre_computed_distance)

        logger.info("Class overrided.")

    def _find_prototypes(self) -> None:
        """Find prototype nodes using the Minimum Spanning Tree (MST) approach."""

        logger.debug("Finding prototypes ...")

        h = Heap(self.subgraph.n_nodes)

        self.subgraph.nodes[0].pred = c.NIL

        h.insert(0)

        prototypes = []
        while not h.is_empty():
            p = h.remove()

            self.subgraph.nodes[p].cost = h.cost[p]

            pred = self.subgraph.nodes[p].pred
            if pred != c.NIL:
                if self.subgraph.nodes[p].label != self.subgraph.nodes[pred].label:
                    if self.subgraph.nodes[p].status != c.PROTOTYPE:
                        self.subgraph.nodes[p].status = c.PROTOTYPE
                        prototypes.append(p)

                    if self.subgraph.nodes[pred].status != c.PROTOTYPE:
                        self.subgraph.nodes[pred].status = c.PROTOTYPE
                        prototypes.append(pred)

            for q in range(self.subgraph.n_nodes):
                if h.color[q] != c.BLACK:
                    if p != q:
                        if self.pre_computed_distance:
                            weight = self.pre_distances[self.subgraph.nodes[p].idx][
                                self.subgraph.nodes[q].idx
                            ]
                        else:
                            weight = self.distance_fn(
                                self.subgraph.nodes[p].features,
                                self.subgraph.nodes[q].features,
                            )

                        if weight < h.cost[q]:
                            self.subgraph.nodes[q].pred = p

                            h.update(q, weight)

        logger.debug("Prototypes: %s.", prototypes)

    def fit(
        self, X_train: np.array, Y_train: np.array, I_train: Optional[np.array] = None
    ) -> None:
        """Fits data in the classifier.

        Args:
            X_train: Array of training features.
            Y_train: Array of training labels.
            I_train: Array of training indexes.

        """

        logger.info("Fitting classifier ...")

        start = time.time()

        self.subgraph = Subgraph(X_train, Y_train, I=I_train)

        self._find_prototypes()

        h = Heap(size=self.subgraph.n_nodes)

        for i in range(self.subgraph.n_nodes):
            if self.subgraph.nodes[i].status == c.PROTOTYPE:
                self.subgraph.nodes[i].pred = c.NIL
                self.subgraph.nodes[i].predicted_label = self.subgraph.nodes[i].label

                h.cost[i] = 0
                h.insert(i)
            else:
                h.cost[i] = c.FLOAT_MAX

        while not h.is_empty():
            p = h.remove()

            self.subgraph.idx_nodes.append(p)
            self.subgraph.nodes[p].cost = h.cost[p]

            for q in range(self.subgraph.n_nodes):
                if p != q:
                    if h.cost[p] < h.cost[q]:
                        if self.pre_computed_distance:
                            weight = self.pre_distances[self.subgraph.nodes[p].idx][
                                self.subgraph.nodes[q].idx
                            ]
                        else:
                            weight = self.distance_fn(
                                self.subgraph.nodes[p].features,
                                self.subgraph.nodes[q].features,
                            )

                        # The current cost will be the maximum cost between the node's and its weight (arc)
                        current_cost = np.maximum(h.cost[p], weight)

                        if current_cost < h.cost[q]:
                            self.subgraph.nodes[q].pred = p
                            self.subgraph.nodes[
                                q
                            ].predicted_label = self.subgraph.nodes[p].predicted_label

                            h.update(q, current_cost)

        self.subgraph.trained = True

        end = time.time()

        train_time = end - start

        logger.info("Classifier has been fitted.")
        logger.info("Training time: %s seconds.", train_time)

    def predict(self, X_val: np.array, I_val: Optional[np.array] = None) -> List[int]:
        """Predicts new data using the pre-trained classifier.

        Args:
            X_val: Array of validation or test features.
            I_val: Array of validation or test indexes.

        Returns:
            (List[int]): A list of predictions for each record of the data.

        """

        if not self.subgraph:
            raise e.BuildError("Subgraph has not been properly created")

        if not self.subgraph.trained:
            raise e.BuildError("Classifier has not been properly fitted")

        logger.info("Predicting data ...")

        start = time.time()

        pred_subgraph = Subgraph(X_val, I=I_val)

        for i in range(pred_subgraph.n_nodes):
            conqueror = -1
            j = 0

            k = self.subgraph.idx_nodes[j]

            if self.pre_computed_distance:
                weight = self.pre_distances[self.subgraph.nodes[k].idx][
                    pred_subgraph.nodes[i].idx
                ]
            else:
                weight = self.distance_fn(
                    self.subgraph.nodes[k].features, pred_subgraph.nodes[i].features
                )

            # The minimum cost will be the maximum between the `k` node cost and its weight (arc)
            min_cost = np.maximum(self.subgraph.nodes[k].cost, weight)

            # The current label will be `k` node's predicted label
            current_label = self.subgraph.nodes[k].predicted_label

            # While `j` is a possible node and the minimum cost is bigger than the current node's cost
            while (
                j < (self.subgraph.n_nodes - 1)
                and min_cost > self.subgraph.nodes[self.subgraph.idx_nodes[j + 1]].cost
            ):
                l = self.subgraph.idx_nodes[j + 1]

                if self.pre_computed_distance:
                    weight = self.pre_distances[self.subgraph.nodes[l].idx][
                        pred_subgraph.nodes[i].idx
                    ]
                else:
                    weight = self.distance_fn(
                        self.subgraph.nodes[l].features, pred_subgraph.nodes[i].features
                    )

                # The temporary minimum cost will be the maximum between the `l` node cost and its weight (arc)
                temp_min_cost = np.maximum(self.subgraph.nodes[l].cost, weight)
                if temp_min_cost < min_cost:
                    min_cost = temp_min_cost
                    conqueror = l
                    current_label = self.subgraph.nodes[l].predicted_label

                j += 1
                k = l

            # Node's `i` predicted label is the same as current label
            pred_subgraph.nodes[i].predicted_label = current_label

            if conqueror > -1:
                self.subgraph.mark_nodes(conqueror)

        preds = [pred.predicted_label for pred in pred_subgraph.nodes]

        end = time.time()

        predict_time = end - start

        logger.info("Data has been predicted.")
        logger.info("Prediction time: %s seconds.", predict_time)

        return preds

    def learn(
        self,
        X_train: np.array,
        Y_train: np.array,
        X_val: np.array,
        Y_val: np.array,
        n_iterations: int = 10,
    ) -> None:
        """Learns the best classifier over a validation set.

        Args:
            X_train: Array of training features.
            Y_train: Array of training labels.
            X_val: Array of validation features.
            Y_val: Array of validation labels.
            n_iterations: Number of iterations.

        """

        logger.info("Learning the best classifier ...")

        max_acc = 0
        previous_acc = 0

        t = 0
        while True:
            logger.info("Running iteration %d/%d ...", t + 1, n_iterations)

            self.fit(X_train, Y_train)

            preds = self.predict(X_val)

            acc = g.opf_accuracy(Y_val, preds)
            if acc > max_acc:
                max_acc = acc
                best_opf = copy.deepcopy(self)
                best_t = t

            errors = np.argwhere(Y_val != preds)

            non_prototypes = 0
            for n in self.subgraph.nodes:
                if n.status != c.PROTOTYPE:
                    non_prototypes += 1

            for err in errors:
                ctr = non_prototypes

                while ctr > 0:
                    j = int(r.generate_uniform_random_number(0, len(X_train)))

                    if self.subgraph.nodes[j].status != c.PROTOTYPE:
                        X_train[j, :], X_val[err, :] = X_val[err, :], X_train[j, :]
                        Y_train[j], Y_val[err] = Y_val[err], Y_train[j]

                        non_prototypes -= 1
                        ctr = 0

                    else:
                        ctr -= 1

            delta = np.fabs(acc - previous_acc)
            previous_acc = acc

            t += 1

            logger.info(
                "Accuracy: %s | Delta: %s | Maximum Accuracy: %s", acc, delta, max_acc
            )

            if delta < 0.0001 or t == n_iterations:
                self = best_opf

                logger.info(
                    "Best classifier has been learned over iteration %d.", best_t + 1
                )

                break

    def prune(
        self,
        X_train: np.array,
        Y_train: np.array,
        X_val: np.array,
        Y_val: np.array,
        n_iterations: int = 10,
    ) -> None:
        """Prunes a classifier over a validation set.

        Args:
            X_train: Array of training features.
            Y_train: Array of training labels.
            X_val: Array of validation features.
            Y_val: Array of validation labels.
            n_iterations: Maximum number of iterations.

        """

        logger.info("Pruning classifier ...")

        self.fit(X_train, Y_train)
        self.predict(X_val)

        initial_nodes = self.subgraph.n_nodes

        for t in range(n_iterations):
            logger.info("Running iteration %d/%d ...", t + 1, n_iterations)

            X_temp, Y_temp = [], []

            # Removes irrelevant nodes
            for j, n in enumerate(self.subgraph.nodes):
                if n.relevant != c.IRRELEVANT:
                    X_temp.append(X_train[j, :])
                    Y_temp.append(Y_train[j])

            X_train = np.asarray(X_temp)
            Y_train = np.asarray(Y_temp)

            self.fit(X_train, Y_train)
            preds = self.predict(X_val)

            acc = g.opf_accuracy(Y_val, preds)

            logger.info("Current accuracy: %s.", acc)

        final_nodes = self.subgraph.n_nodes
        prune_ratio = 1 - final_nodes / initial_nodes

        logger.info("Prune ratio: %s.", prune_ratio)
