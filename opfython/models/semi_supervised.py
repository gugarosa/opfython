"""Semi-Supervised Optimum-Path Forest.
"""

import time
from typing import Optional

import numpy as np

import opfython.utils.constants as c
from opfython.core import Heap, Node, Subgraph
from opfython.models import SupervisedOPF
from opfython.utils import logging

logger = logging.get_logger(__name__)


class SemiSupervisedOPF(SupervisedOPF):
    """A SemiSupervisedOPF which implements the semi-supervised version of OPF classifier.

    References:
        W. P. Amorim, A. X. Falcão and M. H. Carvalho. Semi-supervised Pattern Classification Using Optimum-Path Forest.
        27th SIBGRAPI Conference on Graphics, Patterns and Images (2014).

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

        logger.info("Overriding class: SupervisedOPF -> SemiSupervisedOPF.")

        super(SemiSupervisedOPF, self).__init__(distance, pre_computed_distance)

        logger.info("Class overrided.")

    def fit(
        self,
        X_train: np.array,
        Y_train: np.array,
        X_unlabeled: np.array,
        I_train: Optional[np.array] = None,
    ) -> None:
        """Fits data in the semi-supervised classifier.

        Args:
            X_train: Array of training features.
            Y_train: Array of training labels.
            X_unlabeled: Array of unlabeled features.
            I_train: Array of training indexes.

        """

        logger.info("Fitting semi-supervised classifier ...")

        start = time.time()

        self.subgraph = Subgraph(X_train, Y_train, I_train)

        self._find_prototypes()

        current_n_nodes = self.subgraph.n_nodes
        for i, feature in enumerate(X_unlabeled):
            node = Node(current_n_nodes + i, 0, feature)

            self.subgraph.nodes.append(node)

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

                        current_cost = np.maximum(h.cost[p], weight)
                        if current_cost < h.cost[q]:
                            self.subgraph.nodes[q].pred = p
                            self.subgraph.nodes[
                                q
                            ].predicted_label = self.subgraph.nodes[p].predicted_label

                            # As we may have unlabeled nodes, make sure that `q` label equals to `q` predicted label
                            self.subgraph.nodes[q].label = self.subgraph.nodes[
                                q
                            ].predicted_label

                            h.update(q, current_cost)

        self.subgraph.trained = True

        end = time.time()

        train_time = end - start

        logger.info("Semi-supervised classifier has been fitted.")
        logger.info("Training time: %s seconds.", train_time)
