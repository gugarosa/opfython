"""KNN-based subgraph."""

from typing import Optional

import numpy as np

import opfython.utils.constants as c
import opfython.utils.exception as e
from opfython.core.subgraph import Subgraph
from opfython.utils import logging

logger = logging.get_logger(__name__)


class KNNSubgraph(Subgraph):
    """Subgraph with k-nearest-neighbour adjacency."""

    def __init__(
        self,
        X: Optional[np.array] = None,
        Y: Optional[np.array] = None,
        I: Optional[np.array] = None,
        from_file: Optional[bool] = None,
    ) -> None:
        super().__init__(X, Y, I, from_file)

        self.n_clusters = 0
        self.best_k = 0
        self.constant = 0.0
        self.density = 0.0
        self.min_density = 0.0
        self.max_density = 0.0

    @property
    def n_clusters(self) -> int:
        """Number of clusters."""

        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, n_clusters: int) -> None:
        if not isinstance(n_clusters, int):
            raise e.TypeError("`n_clusters` should be an integer")
        if n_clusters < 0:
            raise e.ValueError("`n_clusters` should be >= 0")
        self._n_clusters = n_clusters

    @property
    def best_k(self) -> int:
        """Selected number of neighbours."""

        return self._best_k

    @best_k.setter
    def best_k(self, best_k: int) -> None:
        if not isinstance(best_k, int):
            raise e.TypeError("`best_k` should be an integer")
        if best_k < 0:
            raise e.ValueError("`best_k` should be >= 0")
        self._best_k = best_k

    @property
    def constant(self) -> float:
        """Probability-density constant."""

        return self._constant

    @constant.setter
    def constant(self, constant: float) -> None:
        if not isinstance(constant, (float, int, np.int32, np.int64)):
            raise e.TypeError("`constant` should be a float or integer")
        self._constant = constant

    @property
    def density(self) -> float:
        """Maximum adjacency distance."""

        return self._density

    @density.setter
    def density(self, density: float) -> None:
        if not isinstance(density, (float, int, np.int32, np.int64)):
            raise e.TypeError("`density` should be a float or integer")
        self._density = density

    @property
    def min_density(self) -> float:
        """Minimum node density."""

        return self._min_density

    @min_density.setter
    def min_density(self, min_density: float) -> None:
        if not isinstance(min_density, (float, int, np.int32, np.int64)):
            raise e.TypeError("`min_density` should be a float or integer")
        self._min_density = min_density

    @property
    def max_density(self) -> float:
        """Maximum node density."""

        return self._max_density

    @max_density.setter
    def max_density(self, max_density: float) -> None:
        if not isinstance(max_density, (float, int, np.int32, np.int64)):
            raise e.TypeError("`max_density` should be a float or integer")
        self._max_density = max_density

    def calculate_pdf(
        self,
        n_neighbours: int,
        distance_function: callable,
        pre_computed_distance: bool = False,
        pre_distances: Optional[np.array] = None,
    ) -> None:
        """Calculate the probability density function for each node."""

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
                (c.MAX_DENSITY - 1)
                * (pdf[i] - self.min_density)
                / (self.max_density - self.min_density)
            ) + 1
            node.cost = node.density - 1

    def create_arcs(
        self,
        k: int,
        distance_function: callable,
        pre_computed_distance: bool = False,
        pre_distances: Optional[np.array] = None,
    ) -> np.array:
        """Create each node's k-nearest-neighbour adjacency."""

        distances = np.zeros(k + 1)
        neighbours_idx = np.zeros(k + 1)
        max_distances = np.zeros(k)

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
        """Reduce node costs by a positive height."""

        logger.debug("Eliminating maxima above height = %s ...", height)
        if height > 0:
            for node in self.nodes:
                node.cost = np.maximum(node.density - height, 0)
        logger.debug("Maxima eliminated.")
