# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Provide the supervised Optimum-Path Forest classifier."""

import copy
import time
from os import PathLike

import numpy as np

import opfython.math.general as g
import opfython.math.random as r
import opfython.utils.constants as c
import opfython.utils.exception as e
from opfython.core.heap import Heap
from opfython.core.opf import OPF
from opfython.core.subgraph import Subgraph
from opfython.models._common import _grow_minimax_forest
from opfython.utils.logging import get_logger

logger = get_logger(__name__)


class SupervisedOPF(OPF):
    """Classify samples with a supervised Optimum-Path Forest on a complete graph."""

    def __init__(
        self,
        distance: str = "log_squared_euclidean",
        pre_computed_distance: str | PathLike[str] | None = None,
    ) -> None:
        """Initialize the distance configuration for supervised training.

        Args:
            distance: Registered distance metric name.
            pre_computed_distance: Optional CSV or text distance-matrix path indexed by sample identifiers.

        Raises:
            opfython.utils.exception.TypeError: The distance metric name is invalid.
            opfython.utils.exception.ArgumentError: The distance file extension is unsupported.
            opfython.utils.exception.ValueError: The distance file cannot be loaded.

        References:
            J. P. Papa, A. X. Falcão and C. T. N. Suzuki.
            Supervised Pattern Classification based on Optimum-Path Forest.
            International Journal of Imaging Systems and Technology (2009).

        """

        logger.info("Overriding class: OPF -> SupervisedOPF.")
        super().__init__(distance, pre_computed_distance)
        logger.info("Class overrided.")

    def _find_prototypes(self) -> None:
        logger.debug("Finding prototypes ...")

        self._validate_pre_distances(self.subgraph)
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
                            weight = self.pre_distances[self.subgraph.nodes[p].idx][self.subgraph.nodes[q].idx]
                        else:
                            weight = self.distance_fn(
                                self.subgraph.nodes[p].features,
                                self.subgraph.nodes[q].features,
                            )

                        if weight < h.cost[q]:
                            self.subgraph.nodes[q].pred = p

                            h.update(q, weight)

        if not prototypes and all(node.label == self.subgraph.nodes[0].label for node in self.subgraph.nodes):
            self.subgraph.nodes[0].status = c.PROTOTYPE
            prototypes.append(0)

        logger.debug("Prototypes: %s.", prototypes)

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        I_train: np.ndarray | None = None,
    ) -> None:
        """Fit a minimax forest from the class-boundary prototypes of a minimum spanning tree.

        Args:
            X_train: Training features with shape (n_samples, n_features), retained without intentional mutation.
            Y_train: Nonnegative training labels with shape (n_samples,), left unchanged.
            I_train: Training distance-matrix indexes, or positional indexes when None.

        Raises:
            opfython.utils.exception.BuildError: Precomputed distances do not cover the training indexes.

        Notes:
            Fitting replaces the stored subgraph and returns None.
            A single-class graph uses its first node as prototype.

        """

        logger.info("Fitting classifier ...")
        start = time.time()

        self.subgraph = Subgraph(X_train, Y_train, I=I_train)
        self._find_prototypes()

        _grow_minimax_forest(self)

        self.subgraph.trained = True

        logger.info("Classifier has been fitted.")
        logger.info("Training time: %s seconds.", time.time() - start)

    def predict(
        self,
        X_val: np.ndarray,
        I_val: np.ndarray | None = None,
    ) -> list[int]:
        """Predicts new data using the pre-trained classifier.

        Args:
            X_val: Query features with shape (n_samples, n_features), left unchanged.
            I_val: Query distance-matrix column indexes, or positional indexes when None.

        Returns:
            Predicted class labels in query order.

        Raises:
            opfython.utils.exception.BuildError: The model is not fitted or the distance matrix misses sample indexes.

        Notes:
            Winning paths are marked relevant on the stored graph for pruning.
            Equal costs retain the first winner in forest order.
            Precomputed distances use training rows and query columns.

        """

        if self.subgraph is None:
            raise e.BuildError("`subgraph` is None; call `fit` before predicting.")

        if not self.subgraph.trained:
            raise e.BuildError("`subgraph.trained` is not True; call `fit` before predicting.")

        logger.info("Predicting data ...")
        start = time.time()
        pred_subgraph = Subgraph(X_val, I=I_val)
        self._validate_pre_distances(self.subgraph, pred_subgraph)

        for i in range(pred_subgraph.n_nodes):
            j = 0

            k = self.subgraph.idx_nodes[j]
            conqueror = k

            if self.pre_computed_distance:
                weight = self.pre_distances[self.subgraph.nodes[k].idx][pred_subgraph.nodes[i].idx]
            else:
                weight = self.distance_fn(self.subgraph.nodes[k].features, pred_subgraph.nodes[i].features)

            min_cost = np.maximum(self.subgraph.nodes[k].cost, weight)

            current_label = self.subgraph.nodes[k].predicted_label

            # Later nodes cannot improve the prediction once their path costs reach the current minimum
            while (
                j < (self.subgraph.n_nodes - 1) and min_cost > self.subgraph.nodes[self.subgraph.idx_nodes[j + 1]].cost
            ):
                l = self.subgraph.idx_nodes[j + 1]

                if self.pre_computed_distance:
                    weight = self.pre_distances[self.subgraph.nodes[l].idx][pred_subgraph.nodes[i].idx]
                else:
                    weight = self.distance_fn(self.subgraph.nodes[l].features, pred_subgraph.nodes[i].features)

                temp_min_cost = np.maximum(self.subgraph.nodes[l].cost, weight)
                if temp_min_cost < min_cost:
                    min_cost = temp_min_cost
                    conqueror = l
                    current_label = self.subgraph.nodes[l].predicted_label

                j += 1

            pred_subgraph.nodes[i].predicted_label = current_label

            self.subgraph.mark_nodes(conqueror)

        preds = [pred.predicted_label for pred in pred_subgraph.nodes]

        logger.info("Data has been predicted.")
        logger.info("Prediction time: %s seconds.", time.time() - start)
        return preds

    def learn(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        n_iterations: int = 10,
    ) -> None:
        """Learns the best classifier over a validation set.

        Args:
            X_train: Training features exchanged in place with misclassified validation features.
            Y_train: Training labels exchanged in place with the corresponding validation labels.
            X_val: Validation features exchanged in place with non-prototype training features.
            Y_val: Validation labels exchanged in place alongside their features.
            n_iterations: Number of iterations.

        Notes:
            Inputs must be writable. The best fitted state is retained, while input exchanges are not rolled back.
            Validation accuracy follows the package's OPF scoring convention.

        """

        logger.info("Learning the best classifier ...")

        max_acc = -np.inf
        best_opf = self
        best_t = 0
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

            errors = np.flatnonzero(Y_val != preds)

            non_prototypes = 0
            for n in self.subgraph.nodes:
                if n.status != c.PROTOTYPE:
                    non_prototypes += 1

            for error_index in errors:
                ctr = non_prototypes

                while ctr > 0:
                    j = int(r.generate_uniform_random_number(0, len(X_train)).item())

                    if self.subgraph.nodes[j].status != c.PROTOTYPE:
                        X_train[j, :], X_val[error_index, :] = (
                            X_val[error_index, :],
                            X_train[j, :].copy(),
                        )
                        Y_train[j], Y_val[error_index] = (
                            Y_val[error_index],
                            Y_train[j],
                        )

                        non_prototypes -= 1
                        ctr = 0

                    else:
                        ctr -= 1

            delta = np.fabs(acc - previous_acc)
            previous_acc = acc

            t += 1

            logger.info(
                "Accuracy: %s | Delta: %s | Maximum Accuracy: %s",
                acc,
                delta,
                max_acc,
            )

            if delta < 0.0001 or t == n_iterations:
                self.__dict__.update(best_opf.__dict__)
                logger.info(
                    "Best classifier has been learned over iteration %d.",
                    best_t + 1,
                )
                break

    def prune(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        n_iterations: int = 10,
    ) -> None:
        """Prunes a classifier over a validation set.

        Args:
            X_train: Training features with shape (n_train, n_features), left unchanged.
            Y_train: Training labels with shape (n_train,), left unchanged.
            X_val: Validation features with shape (n_val, n_features), left unchanged.
            Y_val: Validation labels with shape (n_val,), used for OPF accuracy reporting.
            n_iterations: Maximum number of iterations.

        Notes:
            Each iteration replaces the stored graph with nodes relevant to validation predictions.

        """

        logger.info("Pruning classifier ...")

        self.fit(X_train, Y_train)
        self.predict(X_val)

        initial_nodes = self.subgraph.n_nodes

        for iteration in range(n_iterations):
            logger.info("Running iteration %d/%d ...", iteration + 1, n_iterations)
            X_temp, Y_temp = [], []

            for j, n in enumerate(self.subgraph.nodes):
                if n.relevant != c.IRRELEVANT:
                    X_temp.append(X_train[j, :])
                    Y_temp.append(Y_train[j])

            X_train = np.asarray(X_temp)
            Y_train = np.asarray(Y_temp)

            self.fit(X_train, Y_train)
            preds = self.predict(X_val)

            logger.info("Current accuracy: %s.", g.opf_accuracy(Y_val, preds))

        final_nodes = self.subgraph.n_nodes
        logger.info("Prune ratio: %s.", 1 - final_nodes / initial_nodes)
