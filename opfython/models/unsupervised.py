# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Provide unsupervised Optimum-Path Forest clustering."""

import time
from os import PathLike

import numpy as np

import opfython.utils.constants as c
import opfython.utils.exception as e
from opfython.core.heap import Heap
from opfython.core.opf import OPF
from opfython.models._common import _predict_knn
from opfython.subgraphs.knn import KNNSubgraph
from opfython.utils.logging import get_logger

logger = get_logger(__name__)


class UnsupervisedOPF(OPF):
    """Cluster samples with an unsupervised KNN Optimum-Path Forest."""

    def __init__(
        self,
        min_k: int = 1,
        max_k: int = 1,
        distance: str = "log_squared_euclidean",
        pre_computed_distance: str | PathLike[str] | None = None,
    ) -> None:
        """Initialize the neighbourhood range and distance configuration.

        Args:
            min_k: Minimum `k` value for cutting the subgraph.
            max_k: Maximum `k` value for cutting the subgraph.
            distance: Registered distance metric name.
            pre_computed_distance: Optional CSV or text distance-matrix path indexed by sample identifiers.

        Raises:
            opfython.utils.exception.TypeError: A neighbourhood size is not an integer or the metric name is invalid.
            opfython.utils.exception.ValueError: Neighbourhood bounds are invalid or the distance file cannot be loaded.
            opfython.utils.exception.ArgumentError: The distance file extension is unsupported.

        References:
            L. M. Rocha, F. A. M. Cappabianco, A. X. Falcão.
            Data clustering as an optimum-path forest problem with applications in image analysis.
            International Journal of Imaging Systems and Technology (2009).

        """

        logger.info("Overriding class: OPF -> UnsupervisedOPF.")
        super().__init__(distance, pre_computed_distance)

        self.min_k = min_k
        self.max_k = max_k
        logger.info("Class overrided.")

    @property
    def min_k(self) -> int:
        """Return the positive integer lower bound for neighbourhood selection."""

        return self._min_k

    @min_k.setter
    def min_k(self, min_k: int) -> None:
        if not isinstance(min_k, int):
            raise e.TypeError(f"`min_k` should be an integer, but got {type(min_k).__name__}.")
        if min_k < 1:
            raise e.ValueError(f"`min_k` should be >= 1, but got {min_k}.")

        self._min_k = min_k

    @property
    def max_k(self) -> int:
        """Return the integer upper bound, which must be at least min_k."""

        return self._max_k

    @max_k.setter
    def max_k(self, max_k: int) -> None:
        if not isinstance(max_k, int):
            raise e.TypeError(f"`max_k` should be an integer, but got {type(max_k).__name__}.")
        if max_k < 1:
            raise e.ValueError(f"`max_k` should be >= 1, but got {max_k}.")
        if max_k < self.min_k:
            raise e.ValueError(f"`max_k` should be >= `min_k`, but got {max_k} < {self.min_k}.")

        self._max_k = max_k

    def _clustering(self, n_neighbours: int) -> None:
        self.subgraph.idx_nodes = []

        for i, node in enumerate(self.subgraph.nodes):
            start = node.n_plateaus
            for adjacent in node.adjacency[start : start + n_neighbours]:
                neighbour = self.subgraph.nodes[int(adjacent)]
                if node.density == neighbour.density:
                    end = neighbour.n_plateaus + n_neighbours
                    if i not in neighbour.adjacency[:end]:
                        neighbour.adjacency.insert(0, i)
                        neighbour.n_plateaus += 1

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
                        self.subgraph.nodes[q].cluster_label = self.subgraph.nodes[p].cluster_label

                        h.update(q, current_cost)

        self.subgraph.n_clusters = l

    def _normalized_cut(self, n_neighbours: int) -> float:
        internal_cluster = np.zeros(self.subgraph.n_clusters)
        external_cluster = np.zeros(self.subgraph.n_clusters)

        cut = 0.0

        for i in range(self.subgraph.n_nodes):
            n_adjacents = self.subgraph.nodes[i].n_plateaus + n_neighbours

            for k in range(n_adjacents):
                j = int(self.subgraph.nodes[i].adjacency[k])

                if self.pre_computed_distance:
                    distance = self.pre_distances[self.subgraph.nodes[i].idx][self.subgraph.nodes[j].idx]
                else:
                    distance = self.distance_fn(self.subgraph.nodes[i].features, self.subgraph.nodes[j].features)

                if distance > 0.0:
                    if self.subgraph.nodes[i].cluster_label == self.subgraph.nodes[j].cluster_label:
                        internal_cluster[self.subgraph.nodes[i].cluster_label] += 1 / distance
                    else:
                        external_cluster[self.subgraph.nodes[i].cluster_label] += 1 / distance

        for l in range(self.subgraph.n_clusters):
            if internal_cluster[l] + external_cluster[l] > 0.0:
                cut += external_cluster[l] / (internal_cluster[l] + external_cluster[l])

        return cut

    def _best_minimum_cut(self, min_k: int, max_k: int) -> None:
        logger.debug(
            "Calculating the best minimum cut within [%d, %d] ...",
            min_k,
            max_k,
        )

        max_distances = self.subgraph.create_arcs(
            max_k, self.distance_fn, self.pre_computed_distance, self.pre_distances
        )

        min_cut = c.FLOAT_MAX
        best_k = min_k
        for k in range(min_k, max_k + 1):
            if min_cut != 0.0:
                # Restore the nearest-neighbour order after plateau symmetrization
                for node in self.subgraph.nodes:
                    del node.adjacency[: node.n_plateaus]
                    node.n_plateaus = 0

                self.subgraph.density = max_distances[k - 1]
                if self.subgraph.density < 0.00001:
                    self.subgraph.density = 1
                self.subgraph.best_k = k
                self.subgraph.calculate_pdf(k, self.distance_fn, self.pre_computed_distance, self.pre_distances)

                self._clustering(k)

                cut = self._normalized_cut(k)
                if cut < min_cut:
                    min_cut = cut
                    best_k = k

        self.subgraph.best_k = best_k

        self.subgraph.create_arcs(best_k, self.distance_fn, self.pre_computed_distance, self.pre_distances)
        self.subgraph.calculate_pdf(best_k, self.distance_fn, self.pre_computed_distance, self.pre_distances)

        logger.debug("Best: %d | Minimum cut: %s.", best_k, min_cut)

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray | None = None,
        I_train: np.ndarray | None = None,
    ) -> None:
        """Fit a forest with the neighbourhood that minimizes the normalized cut.

        Args:
            X_train: Training features with shape (n_samples, n_features), retained without intentional mutation.
            Y_train: Optional nonnegative class labels for later root-label propagation.
            I_train: Training distance-matrix indexes, or positional indexes when None.

        Raises:
            opfython.utils.exception.ValueError: The graph cannot supply the configured neighbourhood range.
            opfython.utils.exception.BuildError: Precomputed distances do not cover the training indexes.

        Notes:
            Fitting replaces the stored subgraph and returns None. Cluster identifiers are zero-based.
            Call propagate_labels to assign class predictions from cluster roots.

        """

        logger.info("Clustering with classifier ...")
        start = time.time()

        self.subgraph = KNNSubgraph(X_train, Y_train, I_train)
        if not self.min_k <= self.max_k < self.subgraph.n_nodes:
            raise e.ValueError(
                f"`min_k` and `max_k` should satisfy 1 <= min_k <= max_k < n_nodes, "
                f"but got {self.min_k}, {self.max_k}, and {self.subgraph.n_nodes}."
            )
        self._validate_pre_distances(self.subgraph)
        self._best_minimum_cut(self.min_k, self.max_k)

        self._clustering(self.subgraph.best_k)

        self.subgraph.trained = True

        logger.info("Classifier has been clustered with.")
        logger.info("Number of clusters: %d.", self.subgraph.n_clusters)
        logger.info("Clustering time: %s seconds.", time.time() - start)

    def predict(
        self,
        X_val: np.ndarray,
        I_val: np.ndarray | None = None,
    ) -> tuple[list[int], list[int]]:
        """Predicts new data using the pre-trained classifier.

        Args:
            X_val: Query features with shape (n_samples, n_features), left unchanged.
            I_val: Query distance-matrix row indexes, or positional indexes when None.

        Returns:
            Class-label and zero-based cluster-label lists, both in query order.

        Raises:
            opfython.utils.exception.BuildError: The model is not fitted or the distance matrix misses sample indexes.

        Notes:
            Class labels remain zero until propagate_labels is called.
            Equal distances retain training order, and equal winning costs retain nearest-neighbour order.
            Precomputed distances use query rows and training columns.

        """

        if self.subgraph is None:
            raise e.BuildError("`subgraph` is None; call `fit` before predicting.")

        if not self.subgraph.trained:
            raise e.BuildError("`subgraph.trained` is not True; call `fit` before predicting.")

        logger.info("Predicting data ...")
        start = time.time()
        pred_subgraph = _predict_knn(self, X_val, I_val)
        preds = [pred.predicted_label for pred in pred_subgraph.nodes]
        clusters = [pred.cluster_label for pred in pred_subgraph.nodes]

        logger.info("Data has been predicted.")
        logger.info("Prediction time: %s seconds.", time.time() - start)
        return preds, clusters

    def propagate_labels(self) -> None:
        """Assign each stored node the class label of its cluster root.

        Notes:
            This mutates predicted_label on the fitted graph, not the caller's training labels or cluster assignments.

        """

        logger.info("Assigning predicted labels from clusters ...")

        for i in range(self.subgraph.n_nodes):
            root = self.subgraph.nodes[i].root

            if root == i:
                self.subgraph.nodes[i].predicted_label = self.subgraph.nodes[i].label
            else:
                self.subgraph.nodes[i].predicted_label = self.subgraph.nodes[root].label

        logger.info("Labels assigned.")
