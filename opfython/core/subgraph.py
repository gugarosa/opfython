# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Build and maintain graph state shared by OPF classifiers."""

from os import PathLike
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

import opfython.stream.parser as parser
import opfython.utils.constants as c
import opfython.utils.exception as e
from opfython.core.node import Node
from opfython.stream import loader
from opfython.utils.logging import get_logger

logger = get_logger(__name__)


class Subgraph:
    """Hold samples and their mutable forest relationships."""

    def __init__(
        self,
        X: ArrayLike | None = None,
        Y: np.ndarray | None = None,
        I: np.ndarray | None = None,
        from_file: str | PathLike[str] | None = None,
    ) -> None:
        """Build nodes from arrays or an OPF-formatted dataset.

        Feature storage can remain shared with the input. File rows contain
        an identifier, a label, and feature columns. Parsed file identifiers
        are discarded, so supply I separately when matrix indexes are needed.
        Missing input leaves an empty untrained graph and is logged.

        Args:
            X: Features arranged as samples with their feature values.
            Y: Integer labels with one value per sample, or None to use zero labels.
            I: Nonnegative integer sample indexes used for precomputed matrices.
            from_file: CSV, text, or JSON dataset that replaces X and Y when provided.

        Raises:
            opfython.utils.exception.SizeError: Feature, label, or index counts differ.
            opfython.utils.exception.ArgumentError: The dataset extension is unsupported.
            opfython.utils.exception.TypeError: A node index or label has an invalid type.
            opfython.utils.exception.ValueError: A node value or parsed label sequence is invalid.

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
            logger.error("`X=None` cannot populate a subgraph.")

    @property
    def n_nodes(self) -> int:
        """Return the population derived from the node collection."""

        return len(self.nodes)

    @n_nodes.setter
    def n_nodes(self, n_nodes: int) -> None:
        if not isinstance(n_nodes, int):
            raise e.TypeError(f"`n_nodes` should be an integer, but got {type(n_nodes).__name__}.")
        if n_nodes < 0:
            raise e.ValueError(f"`n_nodes` should be >= 0, but got {n_nodes}.")

        self._n_nodes = n_nodes

    @property
    def n_features(self) -> int:
        """Return the feature count recorded when the graph was built."""

        return self._n_features

    @n_features.setter
    def n_features(self, n_features: int) -> None:
        if not isinstance(n_features, int):
            raise e.TypeError(f"`n_features` should be an integer, but got {type(n_features).__name__}.")
        if n_features < 0:
            raise e.ValueError(f"`n_features` should be >= 0, but got {n_features}.")

        self._n_features = n_features

    @property
    def nodes(self) -> list[Node]:
        """Return the graph's mutable node collection."""

        return self._nodes

    @nodes.setter
    def nodes(self, nodes: list[Node]) -> None:
        if not isinstance(nodes, list):
            raise e.TypeError(f"`nodes` should be a list, but got {type(nodes).__name__}.")

        self._nodes = nodes

    @property
    def idx_nodes(self) -> list[int]:
        """Return graph positions in the current forest's visitation order."""

        return self._idx_nodes

    @idx_nodes.setter
    def idx_nodes(self, idx_nodes: list[int]) -> None:
        if not isinstance(idx_nodes, list):
            raise e.TypeError(f"`idx_nodes` should be a list, but got {type(idx_nodes).__name__}.")

        self._idx_nodes = idx_nodes

    @property
    def trained(self) -> bool:
        """Return whether model training has completed for this graph."""

        return self._trained

    @trained.setter
    def trained(self, trained: bool) -> None:
        if not isinstance(trained, bool):
            raise e.TypeError(f"`trained` should be a boolean, but got {type(trained).__name__}.")

        self._trained = trained

    def _load(self, file_path: str | PathLike[str]) -> tuple[np.ndarray | None, np.ndarray | None]:
        suffix = Path(file_path).suffix.lower()
        loaders = {
            ".csv": loader.load_csv,
            ".json": loader.load_json,
            ".txt": loader.load_txt,
        }

        try:
            load = loaders[suffix]
        except KeyError as error:
            raise e.ArgumentError(f"`file_path` should end with .csv, .json, or .txt, but got {suffix!r}.") from error

        data = load(file_path)
        return parser.parse_loader(data)

    def _build(
        self,
        X: ArrayLike,
        Y: np.ndarray,
        I: np.ndarray | None,
    ) -> None:
        if len(X) != len(Y):
            raise e.SizeError(f"`Y` should have {len(X)} samples to match X, but got {len(Y)}.")
        if I is not None and len(X) != len(I):
            raise e.SizeError(f"`I` should have {len(X)} samples to match X, but got {len(I)}.")

        for index, (features, label) in enumerate(zip(X, Y)):
            if I is not None:
                node = Node(I[index].item(), label.item(), features)
            else:
                node = Node(index, label.item(), features)

            self.nodes.append(node)

        self.n_features = self.nodes[0].features.shape[0] if self.nodes else 0

    def destroy_arcs(self) -> None:
        """Remove every adjacency relation and reset plateau counts."""

        for node in self.nodes:
            node.n_plateaus = 0
            node.adjacency = []

    def mark_nodes(self, i: int) -> None:
        """Mark a node and its predecessor path as relevant to pruning.

        Args:
            i: Position in nodes, not the node's original sample index.

        """

        while self.nodes[i].pred != c.NIL:
            self.nodes[i].relevant = c.RELEVANT
            i = self.nodes[i].pred

        self.nodes[i].relevant = c.RELEVANT

    def reset(self) -> None:
        """Clear predecessors, relevance flags, and adjacency relationships."""

        for node in self.nodes:
            node.pred = c.NIL
            node.relevant = c.IRRELEVANT

        self.destroy_arcs()
