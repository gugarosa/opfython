# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Build KNN adjacency and density state for Optimum-Path Forest models."""

import operator
from collections.abc import Callable
from os import PathLike

import numpy as np
from numpy.typing import ArrayLike

import opfython.utils.constants as c
import opfython.utils.exception as e
from opfython.core.subgraph import Subgraph
from opfython.utils.logging import get_logger

logger = get_logger(__name__)


class KNNSubgraph(Subgraph):
    """Represent a subgraph with k-nearest-neighbour adjacency and density state."""

    def __init__(
        self,
        X: ArrayLike | None = None,
        Y: np.ndarray | None = None,
        I: np.ndarray | None = None,
        from_file: str | PathLike[str] | None = None,
    ) -> None:
        """Build sample nodes with initially empty neighbourhood and density state.

        Args:
            X: Features arranged as (n_samples, n_features), which may share storage with nodes.
            Y: Nonnegative class labels with shape (n_samples,), or None to use zero labels.
            I: Distance-matrix sample indexes, or None to use positional indexes.
            from_file: CSV, text, or JSON dataset replacing X and Y when provided.

        Raises:
            opfython.utils.exception.SizeError: Feature, label, or index counts differ.
            opfython.utils.exception.ArgumentError: The dataset extension is unsupported.

        """

        super().__init__(X, Y, I, from_file)

        self.n_clusters = 0
        self.best_k = 0

        self.constant = 0.0
        self.density = 0.0
        self.min_density = 0.0
        self.max_density = 0.0

    @property
    def n_clusters(self) -> int:
        """Return the nonnegative integer number of clusters."""

        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, n_clusters: int) -> None:
        if not isinstance(n_clusters, int):
            raise e.TypeError(f"`n_clusters` should be an integer, but got {type(n_clusters).__name__}.")
        if n_clusters < 0:
            raise e.ValueError(f"`n_clusters` should be >= 0, but got {n_clusters}.")

        self._n_clusters = n_clusters

    @property
    def best_k(self) -> int:
        """Return the selected nonnegative neighbourhood size, initially zero."""

        return self._best_k

    @best_k.setter
    def best_k(self, best_k: int) -> None:
        if not isinstance(best_k, int):
            raise e.TypeError(f"`best_k` should be an integer, but got {type(best_k).__name__}.")
        if best_k < 0:
            raise e.ValueError(f"`best_k` should be >= 0, but got {best_k}.")

        self._best_k = best_k

    @property
    def constant(self) -> float:
        """Return the numeric probability-density constant, initially zero."""

        return self._constant

    @constant.setter
    def constant(self, constant: float) -> None:
        if not isinstance(constant, (float, int, np.int32, np.int64)):
            raise e.TypeError(f"`constant` should be a float or integer, but got {type(constant).__name__}.")

        self._constant = constant

    @property
    def density(self) -> float:
        """Return the numeric maximum adjacency distance used to scale the density kernel."""

        return self._density

    @density.setter
    def density(self, density: float) -> None:
        if not isinstance(density, (float, int, np.int32, np.int64)):
            raise e.TypeError(f"`density` should be a float or integer, but got {type(density).__name__}.")

        self._density = density

    @property
    def min_density(self) -> float:
        """Return the numeric minimum unscaled node density."""

        return self._min_density

    @min_density.setter
    def min_density(self, min_density: float) -> None:
        if not isinstance(min_density, (float, int, np.int32, np.int64)):
            raise e.TypeError(f"`min_density` should be a float or integer, but got {type(min_density).__name__}.")

        self._min_density = min_density

    @property
    def max_density(self) -> float:
        """Return the numeric maximum unscaled node density."""

        return self._max_density

    @max_density.setter
    def max_density(self, max_density: float) -> None:
        if not isinstance(max_density, (float, int, np.int32, np.int64)):
            raise e.TypeError(f"`max_density` should be a float or integer, but got {type(max_density).__name__}.")

        self._max_density = max_density

    def calculate_pdf(
        self,
        n_neighbours: int,
        distance_function: Callable[[np.ndarray, np.ndarray], float],
        pre_computed_distance: bool = False,
        pre_distances: np.ndarray | None = None,
    ) -> None:
        """Calculate and scale node densities over existing nearest-neighbour arcs.

        Args:
            n_neighbours: Number of existing adjacency entries to use for each node.
            distance_function: Callable receiving two feature arrays and returning their scalar distance.
            pre_computed_distance: Whether to use indexed matrix values instead of the callable.
            pre_distances: Matrix covering all accessed node indexes on both axes, left unchanged.

        Notes:
            Updates graph density bounds and each node's density and initial cost.
            Create arcs first so the density constant and requested adjacency entries are available.
            Equal unscaled densities assign MAX_DENSITY to every node.

        """

        self.constant = 2 * self.density / 9
        self.min_density = c.FLOAT_MAX
        self.max_density = -c.FLOAT_MAX

        pdf = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            n_pdf = 1

            for k in range(n_neighbours):
                j = int(self.nodes[i].adjacency[k])
                if pre_computed_distance:
                    node_distance = pre_distances[self.nodes[i].idx][self.nodes[j].idx]
                else:
                    node_distance = distance_function(
                        self.nodes[i].features,
                        self.nodes[j].features,
                    )

                pdf[i] += np.exp(-node_distance / self.constant)
                n_pdf += 1

            pdf[i] /= n_pdf
            self.min_density = min(self.min_density, pdf[i])
            self.max_density = max(self.max_density, pdf[i])

        if self.min_density == self.max_density:
            for node in self.nodes:
                node.density = c.MAX_DENSITY
                node.cost = c.MAX_DENSITY - 1
            return

        for i, node in enumerate(self.nodes):
            node.density = (
                (c.MAX_DENSITY - 1) * (pdf[i] - self.min_density) / (self.max_density - self.min_density)
            ) + 1
            node.cost = node.density - 1

    def create_arcs(
        self,
        k: int,
        distance_function: Callable[[np.ndarray, np.ndarray], float],
        pre_computed_distance: bool = False,
        pre_distances: np.ndarray | None = None,
    ) -> np.ndarray:
        """Replace each node's adjacency with up to k nearest neighbours.

        Args:
            k: Nonnegative integer-index value, including zero or more neighbours than are available.
            distance_function: Callable receiving two feature arrays and returning their scalar distance.
            pre_computed_distance: Whether to use indexed matrix values instead of the callable.
            pre_distances: Matrix covering all accessed node indexes on both axes, left unchanged.

        Returns:
            Array of shape (k,) with the maximum distance at each neighbour rank, zero for unavailable ranks.

        Raises:
            TypeError: The neighbourhood size does not implement the integer-index protocol.
            ValueError: The neighbourhood size is negative.

        Notes:
            Replaces adjacency, radius, and plateau state without changing caller arrays.
            Equal distances retain node order. Density falls back to one for an effectively zero maximum distance.

        """

        k = operator.index(k)
        distances = np.zeros(k + 1)
        neighbours_idx = np.zeros(k + 1)
        max_distances = np.zeros(k)

        self.destroy_arcs()
        self.density = 0.0

        for i in range(self.n_nodes):
            distances.fill(c.FLOAT_MAX)

            for j in range(self.n_nodes):
                if j == i:
                    continue

                if pre_computed_distance:
                    distances[k] = pre_distances[self.nodes[i].idx][self.nodes[j].idx]
                else:
                    distances[k] = distance_function(
                        self.nodes[i].features,
                        self.nodes[j].features,
                    )

                neighbours_idx[k] = j
                current = k
                while current > 0 and distances[current] < distances[current - 1]:
                    distances[current], distances[current - 1] = (
                        distances[current - 1],
                        distances[current],
                    )
                    neighbours_idx[current], neighbours_idx[current - 1] = (
                        neighbours_idx[current - 1],
                        neighbours_idx[current],
                    )
                    current -= 1

            node = self.nodes[i]
            node.radius = 0.0
            node.n_plateaus = 0

            for neighbour in range(k - 1, -1, -1):
                if distances[neighbour] == c.FLOAT_MAX:
                    continue

                self.density = max(self.density, distances[neighbour])
                node.radius = max(node.radius, distances[neighbour])
                max_distances[neighbour] = max(
                    max_distances[neighbour],
                    distances[neighbour],
                )
                node.adjacency.insert(0, neighbours_idx[neighbour])

        if self.density < 0.00001:
            self.density = 1

        return max_distances

    def eliminate_maxima_height(self, height: float) -> None:
        """Reduce node costs relative to their densities by a positive height.

        Args:
            height: Amount subtracted from each density, with resulting costs bounded below by zero.

        Notes:
            Nonpositive heights leave node costs unchanged.

        """

        logger.debug("Eliminating maxima above height = %s ...", height)
        if height > 0:
            for node in self.nodes:
                node.cost = np.maximum(node.density - height, 0)
        logger.debug("Maxima eliminated.")
