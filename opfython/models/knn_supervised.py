# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Provide the KNN-supervised Optimum-Path Forest classifier."""

import time
from os import PathLike

import numpy as np

import opfython.math.general as g
import opfython.utils.constants as c
import opfython.utils.exception as e
from opfython.core.heap import Heap
from opfython.core.opf import OPF
from opfython.models._common import _predict_knn
from opfython.subgraphs.knn import KNNSubgraph
from opfython.utils.logging import get_logger

logger = get_logger(__name__)


class KNNSupervisedOPF(OPF):
    """Classify samples with a supervised KNN Optimum-Path Forest."""

    def __init__(
        self,
        max_k: int = 1,
        distance: str = "log_squared_euclidean",
        pre_computed_distance: str | PathLike[str] | None = None,
    ) -> None:
        """Initialize the neighbourhood search and distance configuration.

        Args:
            max_k: Maximum `k` value for cutting the subgraph.
            distance: Registered distance metric name.
            pre_computed_distance: Optional CSV or text distance-matrix path indexed by sample identifiers.

        Raises:
            opfython.utils.exception.TypeError: The neighbourhood size is not an integer or the metric name is invalid.
            opfython.utils.exception.ValueError: The neighbourhood bound is less than one or the file cannot be loaded.
            opfython.utils.exception.ArgumentError: The distance file extension is unsupported.

        References:
            J. P. Papa and A. X. Falcão. A Learning Algorithm for the Optimum-Path Forest Classifier.
            Graph-Based Representations in Pattern Recognition (2009).

        """

        logger.info("Overriding class: OPF -> KNNSupervisedOPF.")
        super().__init__(distance, pre_computed_distance)

        self.max_k = max_k
        logger.info("Class overrided.")

    @property
    def max_k(self) -> int:
        """Return the positive integer upper bound for neighbourhood selection."""

        return self._max_k

    @max_k.setter
    def max_k(self, max_k: int) -> None:
        if not isinstance(max_k, int):
            raise e.TypeError(f"`max_k` should be an integer, but got {type(max_k).__name__}.")
        if max_k < 1:
            raise e.ValueError(f"`max_k` should be >= 1, but got {max_k}.")

        self._max_k = max_k

    def _clustering(self, force_prototype: bool = False) -> None:
        self.subgraph.idx_nodes = []

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

                    # The final forest cannot propagate across known class boundaries
                    if force_prototype:
                        if self.subgraph.nodes[p].label != self.subgraph.nodes[q].label:
                            current_cost = -c.FLOAT_MAX

                    if current_cost > h.cost[q]:
                        self.subgraph.nodes[q].pred = p
                        self.subgraph.nodes[q].root = self.subgraph.nodes[p].root
                        self.subgraph.nodes[q].predicted_label = self.subgraph.nodes[p].predicted_label

                        h.update(q, current_cost)

    def _learn(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        I_train: np.ndarray | None,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        I_val: np.ndarray | None,
    ) -> None:
        logger.info("Learning best `k` value ...")

        self.subgraph = KNNSubgraph(X_train, Y_train, I_train)
        if self.max_k >= self.subgraph.n_nodes:
            raise e.ValueError(f"`max_k` should be < `n_nodes`, but got {self.max_k} >= {self.subgraph.n_nodes}.")
        self._validate_pre_distances(self.subgraph)

        max_acc = 0.0
        best_k = 1

        for k in range(1, self.max_k + 1):
            self.subgraph.best_k = k

            self.subgraph.create_arcs(k, self.distance_fn, self.pre_computed_distance, self.pre_distances)
            self.subgraph.calculate_pdf(k, self.distance_fn, self.pre_computed_distance, self.pre_distances)

            self._clustering()

            # Candidate forests are usable internally before the final fit is committed
            predictions = _predict_knn(self, X_val, I_val)
            preds = [node.predicted_label for node in predictions.nodes]

            acc = g.opf_accuracy(Y_val, preds)
            if acc > max_acc:
                max_acc = acc
                best_k = k

            logger.info("Accuracy over k = %d: %s", k, acc)
            self.subgraph.destroy_arcs()

        self.subgraph.best_k = best_k

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        I_train: np.ndarray | None = None,
        I_val: np.ndarray | None = None,
    ) -> None:
        """Fit the forest using the validation accuracy to select its neighbourhood.

        Args:
            X_train: Training features with shape (n_train, n_features), retained without intentional mutation.
            Y_train: Nonnegative training class labels with shape (n_train,).
            X_val: Validation features with shape (n_val, n_features).
            Y_val: Validation class labels with shape (n_val,).
            I_train: Training distance-matrix indexes, or positional indexes when None.
            I_val: Validation distance-matrix indexes, or positional indexes when None.

        Raises:
            opfython.utils.exception.ValueError: The training graph cannot supply the requested neighbourhood.
            opfython.utils.exception.BuildError: Precomputed distances do not cover the accessed sample indexes.

        Notes:
            Fitting replaces the stored subgraph and returns None.
            Accuracy follows the package's OPF scoring convention.

        """

        logger.info("Fitting classifier ...")
        start = time.time()

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

        logger.info("Classifier has been fitted with k = %d.", self.subgraph.best_k)
        logger.info("Training time: %s seconds.", time.time() - start)

    def predict(
        self,
        X_test: np.ndarray,
        I_test: np.ndarray | None = None,
    ) -> list[int]:
        """Predicts new data using the pre-trained classifier.

        Args:
            X_test: Query features with shape (n_samples, n_features), left unchanged.
            I_test: Query distance-matrix row indexes, or positional indexes when None.

        Returns:
            Predicted class labels in query order.

        Raises:
            opfython.utils.exception.BuildError: The model is not fitted or the distance matrix misses sample indexes.

        Notes:
            Equal distances retain training order, and equal winning costs retain nearest-neighbour order.
            Precomputed distances use query rows and training columns.

        """

        if self.subgraph is None:
            raise e.BuildError("`subgraph` is None; call `fit` before predicting.")
        if not self.subgraph.trained:
            raise e.BuildError("`subgraph.trained` is not True; call `fit` before predicting.")

        logger.info("Predicting data ...")
        start = time.time()

        pred_subgraph = _predict_knn(self, X_test, I_test)
        preds = [pred.predicted_label for pred in pred_subgraph.nodes]

        logger.info("Data has been predicted.")
        logger.info("Prediction time: %s seconds.", time.time() - start)
        return preds
