"""Unsupervised Optimum-Path Forest.
"""

import time
from typing import List, Optional

import numpy as np

import opfython.utils.constants as c
import opfython.utils.exception as e
from opfython.core import OPF, Heap
from opfython.subgraphs import KNNSubgraph
from opfython.utils import logging

logger = logging.get_logger(__name__)


class UnsupervisedOPF(OPF):
    """An UnsupervisedOPF which implements the unsupervised version of OPF classifier.

    References:
        L. M. Rocha, F. A. M. Cappabianco, A. X. Falcão.
        Data clustering as an optimum-path forest problem with applications in image analysis.
        International Journal of Imaging Systems and Technology (2009).

    """

    def __init__(
        self,
        min_k: int = 1,
        max_k: int = 1,
        distance: str = "log_squared_euclidean",
        pre_computed_distance: Optional[str] = None,
    ):
        """Initialization method.

        Args:
            min_k: Minimum `k` value for cutting the subgraph.
            max_k: Maximum `k` value for cutting the subgraph.
            distance: An indicator of the distance metric to be used.
            pre_computed_distance: A pre-computed distance file for feeding into OPF.

        """

        logger.info("Overriding class: OPF -> UnsupervisedOPF.")

        super(UnsupervisedOPF, self).__init__(distance, pre_computed_distance)

        self.min_k = min_k
        self.max_k = max_k

        logger.info("Class overrided.")

    @property
    def min_k(self) -> int:
        """Minimum `k` value for cutting the subgraph."""

        return self._min_k

    @min_k.setter
    def min_k(self, min_k: int) -> None:
        if not isinstance(min_k, int):
            raise e.TypeError("`min_k` should be an integer")
        if min_k < 1:
            raise e.ValueError("`min_k` should be >= 1")

        self._min_k = min_k

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
        if max_k < self.min_k:
            raise e.ValueError("`max_k` should be >= `min_k`")

        self._max_k = max_k

    def _clustering(self, n_neighbours: int) -> None:
        """Clusters the subgraph using using a `k` value (number of neighbours).

        Args:
            n_neighbours: Number of neighbours to be used.

        """

        for i in range(self.subgraph.n_nodes):
            for k in range(n_neighbours):
                j = int(self.subgraph.nodes[i].adjacency[k])

                if self.subgraph.nodes[i].density == self.subgraph.nodes[j].density:
                    insert = True

                    for l in range(n_neighbours):
                        adj = int(self.subgraph.nodes[j].adjacency[l])

                        if i == adj:
                            insert = False

                        if insert:
                            self.subgraph.nodes[j].adjacency.insert(0, i)
                            self.subgraph.nodes[j].n_plateaus += 1

        h = Heap(size=self.subgraph.n_nodes, policy="max")

        for i in range(self.subgraph.n_nodes):
            h.cost[i] = self.subgraph.nodes[i].cost

            self.subgraph.nodes[i].pred = c.NIL
            self.subgraph.nodes[i].root = i

            h.insert(i)

        l = 0
        while not h.is_empty():
            p = h.remove()

            self.subgraph.idx_nodes.append(p)

            if self.subgraph.nodes[p].pred == c.NIL:
                h.cost[p] = self.subgraph.nodes[p].density

                self.subgraph.nodes[p].cluster_label = l
                l += 1

            self.subgraph.nodes[p].cost = h.cost[p]

            n_adjacents = self.subgraph.nodes[p].n_plateaus + n_neighbours
            for k in range(n_adjacents):
                q = int(self.subgraph.nodes[p].adjacency[k])

                if h.color[q] != c.BLACK:
                    current_cost = np.minimum(h.cost[p], self.subgraph.nodes[q].density)

                    if current_cost > h.cost[q]:
                        self.subgraph.nodes[q].pred = p
                        self.subgraph.nodes[q].root = self.subgraph.nodes[p].root
                        self.subgraph.nodes[q].cluster_label = self.subgraph.nodes[
                            p
                        ].cluster_label

                        h.update(q, current_cost)

        # The final number of clusters will be equal to `l`
        self.subgraph.n_clusters = l

    def _normalized_cut(self, n_neighbours: int) -> int:
        """Performs a normalized cut over the subgraph using a `k` value (number of neighbours).

        Args:
            n_neighbours: Number of neighbours to be used.

        Returns:
            (int): The value of the normalized cut.

        """

        internal_cluster = np.zeros(self.subgraph.n_clusters)
        external_cluster = np.zeros(self.subgraph.n_clusters)

        cut = 0.0

        for i in range(self.subgraph.n_nodes):
            n_adjacents = self.subgraph.nodes[i].n_plateaus + n_neighbours

            for k in range(n_adjacents):
                j = int(self.subgraph.nodes[i].adjacency[k])

                if self.pre_computed_distance:
                    distance = self.pre_distances[self.subgraph.nodes[i].idx][
                        self.subgraph.nodes[j].idx
                    ]
                else:
                    distance = self.distance_fn(
                        self.subgraph.nodes[i].features, self.subgraph.nodes[j].features
                    )

                if distance > 0.0:
                    if (
                        self.subgraph.nodes[i].cluster_label
                        == self.subgraph.nodes[j].cluster_label
                    ):
                        internal_cluster[self.subgraph.nodes[i].cluster_label] += (
                            1 / distance
                        )
                    else:
                        external_cluster[self.subgraph.nodes[i].cluster_label] += (
                            1 / distance
                        )

        for l in range(self.subgraph.n_clusters):
            if internal_cluster[l] + external_cluster[l] > 0.0:
                cut += external_cluster[l] / (internal_cluster[l] + external_cluster[l])

        return cut

    def _best_minimum_cut(self, min_k: int, max_k: int) -> None:
        """Performs a minimum cut on the subgraph using the best `k` value.

        Args:
            min_k: Minimum value of k.
            max_k: Maximum value of k.

        """

        logger.debug(
            "Calculating the best minimum cut within [%d, %d] ...", min_k, max_k
        )

        max_distances = self.subgraph.create_arcs(
            max_k, self.distance_fn, self.pre_computed_distance, self.pre_distances
        )

        min_cut = c.FLOAT_MAX
        for k in range(min_k, max_k + 1):
            if min_cut != 0.0:
                self.subgraph.density = max_distances[k - 1]
                self.subgraph.best_k = k
                self.subgraph.calculate_pdf(
                    k, self.distance_fn, self.pre_computed_distance, self.pre_distances
                )

                self._clustering(k)

                cut = self._normalized_cut(k)
                if cut < min_cut:
                    min_cut = cut
                    best_k = k

        self.subgraph.destroy_arcs()

        self.subgraph.best_k = best_k

        self.subgraph.create_arcs(
            best_k, self.distance_fn, self.pre_computed_distance, self.pre_distances
        )
        self.subgraph.calculate_pdf(
            best_k, self.distance_fn, self.pre_computed_distance, self.pre_distances
        )

        logger.debug("Best: %d | Minimum cut: %d.", best_k, min_cut)

    def fit(
        self,
        X_train: np.array,
        Y_train: Optional[np.array] = None,
        I_train: Optional[np.array] = None,
    ) -> None:
        """Fits data in the classifier.

        Args:
            X_train: Array of training features.
            Y_train: Array of training labels.
            I_train: Array of training indexes.

        """

        logger.info("Clustering with classifier ...")

        start = time.time()

        self.subgraph = KNNSubgraph(X_train, Y_train, I_train)

        self._best_minimum_cut(self.min_k, self.max_k)

        self._clustering(self.subgraph.best_k)

        self.subgraph.trained = True

        end = time.time()

        train_time = end - start

        logger.info("Classifier has been clustered with.")
        logger.info("Number of clusters: %d.", self.subgraph.n_clusters)
        logger.info("Clustering time: %s seconds.", train_time)

    def predict(self, X_val: np.array, I_val: Optional[np.array] = None) -> List[int]:
        """Predicts new data using the pre-trained classifier.

        Args:
            X_val: Array of validation features.
            I_val: Array of validation indexes.

        Returns:
            (List[int]): A list of predictions for each record of the data.

        """

        if not self.subgraph:
            raise e.BuildError("KNNSubgraph has not been properly created")

        if not self.subgraph.trained:
            raise e.BuildError("Classifier has not been properly clustered")

        logger.info("Predicting data ...")

        start = time.time()

        pred_subgraph = KNNSubgraph(X_val, I=I_val)

        best_k = self.subgraph.best_k

        distances = np.zeros(best_k + 1)
        neighbours_idx = np.zeros(best_k + 1)

        for i in range(pred_subgraph.n_nodes):
            cost = -c.FLOAT_MAX
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

            # Scale the density between minimum and maximum values
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

                        # Propagates the predicted label from the neighbour
                        pred_subgraph.nodes[i].predicted_label = self.subgraph.nodes[
                            neighbour
                        ].predicted_label

                        # Propagates the cluster label from the neighbour
                        pred_subgraph.nodes[i].cluster_label = self.subgraph.nodes[
                            neighbour
                        ].cluster_label

        preds = [pred.predicted_label for pred in pred_subgraph.nodes]
        clusters = [pred.cluster_label for pred in pred_subgraph.nodes]

        end = time.time()

        predict_time = end - start

        logger.info("Data has been predicted.")
        logger.info("Prediction time: %s seconds.", predict_time)

        return preds, clusters

    def propagate_labels(self) -> None:
        """Runs through the clusters and propagate the clusters roots labels to the samples."""

        logger.info("Assigning predicted labels from clusters ...")

        for i in range(self.subgraph.n_nodes):
            root = self.subgraph.nodes[i].root

            if root == i:
                self.subgraph.nodes[i].predicted_label = self.subgraph.nodes[i].label
            else:
                self.subgraph.nodes[i].predicted_label = self.subgraph.nodes[root].label

        logger.info("Labels assigned.")
