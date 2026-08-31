"""Subgraph structure that belongs to the Optimum-Path Forest."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import opfython.stream.parser as parser
import opfython.utils.constants as c
import opfython.utils.exception as e
from opfython.core.node import Node
from opfython.stream import loader
from opfython.utils import logging

logger = logging.get_logger(__name__)


class Subgraph:
    """Collection of nodes used by OPF classifiers."""

    def __init__(
        self,
        X: Optional[np.array] = None,
        Y: Optional[np.array] = None,
        I: Optional[np.array] = None,
        from_file: Optional[bool] = None,
    ) -> None:
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
        """Graph nodes."""

        return self._nodes

    @nodes.setter
    def nodes(self, nodes: List[Node]) -> None:
        if not isinstance(nodes, list):
            raise e.TypeError("`nodes` should be a list")
        self._nodes = nodes

    @property
    def idx_nodes(self) -> List[int]:
        """Nodes ordered by path cost."""

        return self._idx_nodes

    @idx_nodes.setter
    def idx_nodes(self, idx_nodes: List[int]) -> None:
        if not isinstance(idx_nodes, list):
            raise e.TypeError("`idx_nodes` should be a list")
        self._idx_nodes = idx_nodes

    @property
    def trained(self) -> bool:
        """Whether the graph has been trained."""

        return self._trained

    @trained.setter
    def trained(self, trained: bool) -> None:
        if not isinstance(trained, bool):
            raise e.TypeError("`trained` should be a boolean")
        self._trained = trained

    def _load(self, file_path: str) -> Tuple[np.array, np.array]:
        """Load and parse a dataset."""

        suffix = Path(file_path).suffix.lower()
        loaders = {
            ".csv": loader.load_csv,
            ".json": loader.load_json,
            ".txt": loader.load_txt,
        }
        try:
            load = loaders[suffix]
        except KeyError as error:
            raise e.ArgumentError(
                "File extension should be `.csv`, `.json`, or `.txt`"
            ) from error

        data = load(file_path)
        return parser.parse_loader(data)

    def _build(
        self,
        X: np.array,
        Y: np.array,
        I: np.array,
    ) -> None:
        """Build nodes from feature, label, and optional index arrays."""

        for index, (features, label) in enumerate(zip(X, Y)):
            if I is not None:
                node = Node(I[index].item(), label.item(), features)
            else:
                node = Node(index, label.item(), features)

            self.nodes.append(node)
        self.n_features = self.nodes[0].features.shape[0] if self.nodes else 0

    def destroy_arcs(self) -> None:
        """Destroy every adjacency relation."""

        for node in self.nodes:
            node.n_plateaus = 0
            node.adjacency = []

    def mark_nodes(self, i: int) -> None:
        """Mark a node and its predecessor path as relevant."""

        while self.nodes[i].pred != c.NIL:
            self.nodes[i].relevant = c.RELEVANT
            i = self.nodes[i].pred

        self.nodes[i].relevant = c.RELEVANT

    def reset(self) -> None:
        """Reset predecessors, relevance flags, and arcs."""

        for node in self.nodes:
            node.pred = c.NIL
            node.relevant = c.IRRELEVANT

        self.destroy_arcs()
