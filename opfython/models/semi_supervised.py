# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Provide the semi-supervised Optimum-Path Forest classifier."""

import time
from os import PathLike

import numpy as np

from opfython.core.node import Node
from opfython.core.subgraph import Subgraph
from opfython.models._common import _grow_minimax_forest
from opfython.models.supervised import SupervisedOPF
from opfython.utils.logging import get_logger

logger = get_logger(__name__)


class SemiSupervisedOPF(SupervisedOPF):
    """Classify labelled and unlabelled samples with a semi-supervised Optimum-Path Forest."""

    def __init__(
        self,
        distance: str = "log_squared_euclidean",
        pre_computed_distance: str | PathLike[str] | None = None,
    ) -> None:
        """Initialize the distance configuration for semi-supervised training.

        Args:
            distance: Registered distance metric name.
            pre_computed_distance: Optional CSV or text distance-matrix path indexed by sample identifiers.

        Raises:
            opfython.utils.exception.TypeError: The distance metric name is invalid.
            opfython.utils.exception.ArgumentError: The distance file extension is unsupported.
            opfython.utils.exception.ValueError: The distance file cannot be loaded.

        References:
            W. P. Amorim, A. X. Falcão and M. H. Carvalho.
            Semi-supervised Pattern Classification Using Optimum-Path Forest.
            27th SIBGRAPI Conference on Graphics, Patterns and Images (2014).

        """

        logger.info("Overriding class: SupervisedOPF -> SemiSupervisedOPF.")
        super().__init__(distance, pre_computed_distance)
        logger.info("Class overrided.")

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_unlabeled: np.ndarray,
        I_train: np.ndarray | None = None,
    ) -> None:
        """Grow a forest over labelled and unlabelled samples from labelled prototypes.

        Args:
            X_train: Labelled features with shape (n_train, n_features), retained without intentional mutation.
            Y_train: Nonnegative training labels with shape (n_train,), left unchanged.
            X_unlabeled: Unlabelled features whose distance-matrix indexes start at len(X_train), in array order.
            I_train: Labelled distance-matrix indexes, or positional indexes when None.

        Raises:
            opfython.utils.exception.BuildError: Precomputed distances do not cover labelled and unlabelled indexes.

        Notes:
            Fitting replaces the stored subgraph and returns None. Successful relaxations update both label and
            predicted_label on graph nodes, including labelled non-prototypes, without modifying Y_train.
            Unlabelled indexes do not depend on the values in I_train.

        """

        logger.info("Fitting semi-supervised classifier ...")
        start = time.time()

        self.subgraph = Subgraph(X_train, Y_train, I_train)
        self._find_prototypes()

        current_n_nodes = self.subgraph.n_nodes
        for i, feature in enumerate(X_unlabeled):
            node = Node(current_n_nodes + i, 0, feature)

            self.subgraph.nodes.append(node)

        self._validate_pre_distances(self.subgraph)
        _grow_minimax_forest(self, update_labels=True)

        self.subgraph.trained = True

        logger.info("Semi-supervised classifier has been fitted.")
        logger.info("Training time: %s seconds.", time.time() - start)
