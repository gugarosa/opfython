"""KNN-Supervised Optimum-Path Forest.
"""

import time
from typing import List, Optional

import numpy as np

import opfython.math.general as g
import opfython.utils.constants as c
import opfython.utils.exception as e
from opfython.core import OPF, Heap
from opfython.subgraphs import KNNSubgraph
from opfython.utils import logging

logger = logging.get_logger(__name__)


class KNNSupervisedOPF(OPF):
    """A KNNSupervisedOPF which implements the supervised version of OPF classifier with a KNN subgraph.

    References:
        J. P. Papa and A. X. Falcão. A Learning Algorithm for the Optimum-Path Forest Classifier.
        Graph-Based Representations in Pattern Recognition (2009).

    """

    def __init__(
        self,
        max_k: int = 1,
        distance: Optional[str] = "log_squared_euclidean",
        pre_computed_distance: Optional[str] = None,
    ) -> None:
        """Initialization method.

        Args:
            max_k: Maximum `k` value for cutting the subgraph.
            distance: An indicator of the distance metric to be used.
            pre_computed_distance: A pre-computed distance file for feeding into OPF.

        """

        logger.info("Overriding class: OPF -> KNNSupervisedOPF.")

        super(KNNSupervisedOPF, self).__init__(distance, pre_computed_distance)

        self.max_k = max_k

        logger.info("Class overrided.")

    @property
    def max_k(self) -> int:
        """Maximum `k` value for cutting the subgraph."""

        return self._max_k

    @max_k.setter
    def max_k(self, max_k: int) -> None:
        if not isinstance(max_k, int):
            raise e.TypeError("`max_k` should be an integer")
        if max_k < 1:
            raise e.ValueError("`max_k` should be >= 1")

        self._max_k = max_k

    def _clustering(self, force_prototype: bool = False) -> None:
        """Clusters the subgraph.

        Args:
            force_prototype: Whether clustering should for each class to have at least one prototype.

        """

        for i in range(self.subgraph.n_nodes):
            for j in self.subgraph.nodes[i].adjacency:
                j = int(j)

                if self.subgraph.nodes[i].density == self.subgraph.nodes[j].density:
                    insert = True

                    for l in self.subgraph.nodes[j].adjacency:
                        l = int(l)

                        if i == l:
                            insert = False

                    if insert:
                        self.subgraph.nodes[j].adjacency.insert(0, i)

        h = Heap(size=self.subgraph.n_nodes, policy="max")

        for i in range(self.subgraph.n_nodes):
            h.cost[i] = self.subgraph.nodes[i].cost

            self.subgraph.nodes[i].pred = c.NIL
            self.subgraph.nodes[i].root = i

            h.insert(i)

        while not h.is_empty():
            p = h.remove()

            self.subgraph.idx_nodes.append(p)

            if self.subgraph.nodes[p].pred == c.NIL:
                h.cost[p] = self.subgraph.nodes[p].density
                self.subgraph.nodes[p].predicted_label = self.subgraph.nodes[p].label

            self.subgraph.nodes[p].cost = h.cost[p]

            for q in self.subgraph.nodes[p].adjacency:
                q = int(q)

                if h.color[q] != c.BLACK:
                    current_cost = np.minimum(h.cost[p], self.subgraph.nodes[q].density)

                    # If prototypes should be forced to belong to a class
                    if force_prototype:
                        if self.subgraph.nodes[p].label != self.subgraph.nodes[q].label:
                            current_cost = -c.FLOAT_MAX

                    if current_cost > h.cost[q]:
                        self.subgraph.nodes[q].pred = p
                        self.subgraph.nodes[q].root = self.subgraph.nodes[p].root
                        self.subgraph.nodes[q].predicted_label = self.subgraph.nodes[
                            p
                        ].predicted_label

                        h.update(q, current_cost)

    def _learn(
        self,
        X_train: np.array,
        Y_train: np.array,
        I_train: np.array,
        X_val: np.array,
        Y_val: np.array,
        I_val: np.array,
    ) -> None:
        """Learns the best `k` value over the validation set.

        Args:
            X_train: Array of training features.
            Y_train: Array of training labels.
            I_train: Array of training indexes.
            X_val: Array of validation features.
            Y_val: Array of validation labels.
            I_val: Array of validation indexes.

        """

        logger.info("Learning best `k` value ...")

        self.subgraph = KNNSubgraph(X_train, Y_train, I_train)

        if self.pre_computed_distance:
            if (
                self.pre_distances.shape[0] != self.subgraph.n_nodes
                or self.pre_distances.shape[1] != self.subgraph.n_nodes
            ):
                raise e.BuildError(
                    "Pre-computed distance matrix should have the size of `n_nodes x n_nodes`"
                )

        max_acc = 0.0

        for k in range(1, self.max_k + 1):
            self.subgraph.best_k = k

            self.subgraph.create_arcs(
                k, self.distance_fn, self.pre_computed_distance, self.pre_distances
            )
            self.subgraph.calculate_pdf(
                k, self.distance_fn, self.pre_computed_distance, self.pre_distances
            )

            self._clustering()

            preds = self.predict(X_val, I_val)

            acc = g.opf_accuracy(Y_val, preds)
            if acc > max_acc:
                max_acc = acc
                best_k = k

            logger.info("Accuracy over k = %d: %s", k, acc)

            self.subgraph.destroy_arcs()

        self.subgraph.best_k = best_k

    def fit(
        self,
        X_train: np.array,
        Y_train: np.array,
        X_val: np.array,
        Y_val: np.array,
        I_train: Optional[np.array] = None,
        I_val: Optional[np.array] = None,
    ) -> None:
        """Fits data in the classifier.

        Args:
            X_train: Array of training features.
            Y_train: Array of training labels.
            X_val: Array of validation features.
            Y_val: Array of validation labels.
            I_train: Array of training indexes.
            I_val: Array of validation indexes.

        """

        logger.info("Fitting classifier ...")

        start = time.time()

        # Performing the learning process in order to find the best `k` value
        self._learn(X_train, Y_train, I_train, X_val, Y_val, I_val)

        self.subgraph.create_arcs(
            self.subgraph.best_k,
            self.distance_fn,
            self.pre_computed_distance,
            self.pre_distances,
        )
        self.subgraph.calculate_pdf(
            self.subgraph.best_k,
            self.distance_fn,
            self.pre_computed_distance,
            self.pre_distances,
        )

        self._clustering(force_prototype=True)

        self.subgraph.destroy_arcs()

        self.subgraph.trained = True

        end = time.time()

        train_time = end - start

        logger.info("Classifier has been fitted with k = %d.", self.subgraph.best_k)
        logger.info("Training time: %s seconds.", train_time)

    def predict(self, X_test: np.array, I_test: Optional[np.array] = None) -> List[int]:
        """Predicts new data using the pre-trained classifier.

        Args:
            X_test: Array of features.
            I_test: Array of indexes.

        Returns:
            (List[int]): A list of predictions for each record of the data.

        """

        logger.info("Predicting data ...")

        start = time.time()

        pred_subgraph = KNNSubgraph(X_test, I=I_test)

        best_k = self.subgraph.best_k

        distances = np.zeros(best_k + 1)
        neighbours_idx = np.zeros(best_k + 1)

        for i in range(pred_subgraph.n_nodes):
            cost = c.FLOAT_MAX * -1

            distances.fill(c.FLOAT_MAX)

            for j in range(self.subgraph.n_nodes):
                if j != i:
                    if self.pre_computed_distance:
                        distances[best_k] = self.pre_distances[
                            pred_subgraph.nodes[i].idx
                        ][self.subgraph.nodes[j].idx]
                    else:
                        distances[best_k] = self.distance_fn(
                            pred_subgraph.nodes[i].features,
                            self.subgraph.nodes[j].features,
                        )

                    neighbours_idx[best_k] = j
                    cur_k = best_k

                    # While current `k` is bigger than 0 and the `k` distance is smaller than `k-1` distance
                    while cur_k > 0 and distances[cur_k] < distances[cur_k - 1]:
                        distances[cur_k], distances[cur_k - 1] = (
                            distances[cur_k - 1],
                            distances[cur_k],
                        )

                        neighbours_idx[cur_k], neighbours_idx[cur_k - 1] = (
                            neighbours_idx[cur_k - 1],
                            neighbours_idx[cur_k],
                        )

                        cur_k -= 1

            density = 0.0
            for k in range(best_k):
                density += np.exp(-distances[k] / self.subgraph.constant)
            density /= best_k

            density = (
                (c.MAX_DENSITY - 1)
                * (density - self.subgraph.min_density)
                / (self.subgraph.max_density - self.subgraph.min_density + c.EPSILON)
            ) + 1

            for k in range(best_k):
                if distances[k] != c.FLOAT_MAX:
                    neighbour = int(neighbours_idx[k])

                    temp_cost = np.minimum(self.subgraph.nodes[neighbour].cost, density)
                    if temp_cost > cost:
                        cost = temp_cost

                        pred_subgraph.nodes[i].predicted_label = self.subgraph.nodes[
                            neighbour
                        ].predicted_label

        preds = [pred.predicted_label for pred in pred_subgraph.nodes]

        end = time.time()

        predict_time = end - start

        logger.info("Data has been predicted.")
        logger.info("Prediction time: %s seconds.", predict_time)

        return preds
