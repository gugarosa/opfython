"""Subgraph structure that belongs to the Optimum-Path Forest.
"""

from typing import List, Optional, Tuple

import numpy as np

import opfython.stream.parser as p
import opfython.utils.constants as c
import opfython.utils.exception as e
from opfython.core import Node
from opfython.stream import loader
from opfython.utils import logging

logger = logging.get_logger(__name__)


class Subgraph:
    """A Subgraph class is used as a collection of Nodes and the basic structure to work with OPF."""

    def __init__(
        self,
        X: Optional[np.array] = None,
        Y: Optional[np.array] = None,
        I: Optional[np.array] = None,
        from_file: Optional[bool] = None,
    ) -> None:
        """Initialization method.

        Args:
            X: Array of features.
            Y: Array of labels.
            I: Array of indexes.
            from_file: Whether Subgraph should be directly created from a file.

        """

        self.n_nodes = 0
        self.n_features = 0

        self.nodes = []
        self.idx_nodes = []

        self.trained = False

        if from_file:
            X, Y = self._load(from_file)

        if X is not None:
            if Y is None:
                Y = np.zeros(len(X), dtype=int)

            self._build(X, Y, I)

        else:
            logger.error("Subgraph has not been properly created.")

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""

        return len(self.nodes)

    @n_nodes.setter
    def n_nodes(self, n_nodes: int) -> None:
        if not isinstance(n_nodes, int):
            raise e.TypeError("`n_nodes` should be an integer")
        if n_nodes < 0:
            raise e.ValueError("`n_nodes` should be >= 0")

        self._n_nodes = n_nodes

    @property
    def n_features(self) -> int:
        """Number of features."""

        return self._n_features

    @n_features.setter
    def n_features(self, n_features: int) -> None:
        if not isinstance(n_features, int):
            raise e.TypeError("`n_features` should be an integer")
        if n_features < 0:
            raise e.ValueError("`n_features` should be >= 0")

        self._n_features = n_features

    @property
    def nodes(self) -> List[Node]:
        """List of nodes that belongs to the Subgraph."""

        return self._nodes

    @nodes.setter
    def nodes(self, nodes: List[Node]) -> None:
        if not isinstance(nodes, list):
            raise e.TypeError("`nodes` should be a list")

        self._nodes = nodes

    @property
    def idx_nodes(self) -> List[int]:
        """List of ordered nodes indexes."""

        return self._idx_nodes

    @idx_nodes.setter
    def idx_nodes(self, idx_nodes: List[int]) -> None:
        if not isinstance(idx_nodes, list):
            raise e.TypeError("`idx_nodes` should be a list")

        self._idx_nodes = idx_nodes

    @property
    def trained(self) -> bool:
        """Indicate whether the subgraph is trained."""

        return self._trained

    @trained.setter
    def trained(self, trained: bool) -> None:
        if not isinstance(trained, bool):
            raise e.TypeError("`trained` should be a boolean")

        self._trained = trained

    def _load(self, file_path: str) -> Tuple[np.array, np.array]:
        """Loads and parses a dataframe from a file.

        Args:
            file_path: File to be loaded.

        Returns:
            (Tuple[np.array, np.array]): Arrays holding the features and labels.

        """

        extension = file_path.split(".")[-1]

        if extension == "csv":
            data = loader.load_csv(file_path)
        elif extension == "txt":
            data = loader.load_txt(file_path)
        elif extension == "json":
            data = loader.load_json(file_path)
        else:
            raise e.ArgumentError(
                "File extension not recognized. It should be `.csv`, `.json` or `.txt`"
            )

        X, Y = p.parse_loader(data)

        return X, Y

    def _build(self, X: np.array, Y: np.array, I: np.array) -> None:
        """This method serves as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            X: Features array.
            Y: Labels array.
            I: Indexes array.

        """

        for i, (feature, label) in enumerate(zip(X, Y)):
            if I is not None:
                node = Node(I[i].item(), label.item(), feature)
            else:
                node = Node(i, label.item(), feature)

            self.nodes.append(node)

        self.n_features = self.nodes[0].features.shape[0]

    def destroy_arcs(self) -> None:
        """Destroy the arcs present in the subgraph."""

        for i in range(self.n_nodes):
            self.nodes[i].n_plateaus = 0
            self.nodes[i].adjacency = []

    def mark_nodes(self, i: int) -> None:
        """Marks a node and its whole path as relevant.

        Args:
            i: An identifier of the node to start the marking.

        """

        while self.nodes[i].pred != c.NIL:
            self.nodes[i].relevant = c.RELEVANT
            i = self.nodes[i].pred

        self.nodes[i].relevant = c.RELEVANT

    def reset(self) -> None:
        """Resets the subgraph predecessors and arcs."""

        for i in range(self.n_nodes):
            self.nodes[i].pred = c.NIL
            self.nodes[i].relevant = c.IRRELEVANT

        self.destroy_arcs()
