"""KNN-based Subgraph.
"""

from typing import Optional

import numpy as np

import opfython.utils.constants as c
import opfython.utils.exception as e
from opfython.core import Subgraph
from opfython.utils import logging

logger = logging.get_logger(__name__)


class KNNSubgraph(Subgraph):
    """A KNNSubgraph is used to implement a k-nearest neightbours subgraph."""

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

        super(KNNSubgraph, self).__init__(X, Y, I, from_file)

        self.n_clusters = 0
        self.best_k = 0
        self.constant = 0.0

        self.density = 0.0
        self.min_density = 0.0
        self.max_density = 0.0

    @property
    def n_clusters(self) -> int:
        """int: Number of assigned clusters."""

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
        """int: Number of adjacent nodes (k-nearest neighbours)."""

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
        """float: Constant used to calculate the probability density function."""

        return self._constant

    @constant.setter
    def constant(self, constant: float) -> None:
        if not isinstance(constant, (float, int, np.int32, np.int64)):
            raise e.TypeError("`constant` should be a float or integer")

        self._constant = constant

    @property
    def density(self) -> float:
        """float: Density of the subgraph."""

        return self._density

    @density.setter
    def density(self, density: float) -> None:
        if not isinstance(density, (float, int, np.int32, np.int64)):
            raise e.TypeError("`density` should be a float or integer")

        self._density = density

    @property
    def min_density(self) -> float:
        """float: Minimum density of the subgraph."""

        return self._min_density

    @min_density.setter
    def min_density(self, min_density: float) -> None:
        if not isinstance(min_density, (float, int, np.int32, np.int64)):
            raise e.TypeError("`min_density` should be a float or integer")

        self._min_density = min_density

    @property
    def max_density(self) -> float:
        """float: Maximum density of the subgraph."""

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
        """Calculates the probability density function for `k` neighbours.

        Args:
            n_neighbours: Number of neighbours in the adjacency relation.
            distance_function: The distance function to be used to calculate the arcs.
            pre_computed_distance: Whether OPF should use a pre-computed distance or not.
            pre_distances: Pre-computed distance matrix.

        """

        self.constant = 2 * self.density / 9

        self.min_density = c.FLOAT_MAX
        self.max_density = -c.FLOAT_MAX

        pdf = np.zeros(self.n_nodes)
        for i in range(self.n_nodes):
            pdf[i] = 0
            n_pdf = 1

            for k in range(n_neighbours):
                j = int(self.nodes[i].adjacency[k])

                if pre_computed_distance:
                    distance = pre_distances[self.nodes[i].idx][self.nodes[j].idx]

                else:
                    distance = distance_function(
                        self.nodes[i].features, self.nodes[j].features
                    )

                pdf[i] += np.exp(-distance / self.constant)
                n_pdf += 1

            pdf[i] /= n_pdf

            if pdf[i] < self.min_density:
                self.min_density = pdf[i]
            if pdf[i] > self.max_density:
                self.max_density = pdf[i]

        if self.min_density == self.max_density:
            for i in range(self.n_nodes):
                self.nodes[i].density = c.MAX_DENSITY
                self.nodes[i].cost = c.MAX_DENSITY - 1
        else:
            for i in range(self.n_nodes):
                self.nodes[i].density = (
                    (c.MAX_DENSITY - 1)
                    * (pdf[i] - self.min_density)
                    / (self.max_density - self.min_density)
                ) + 1
                self.nodes[i].cost = self.nodes[i].density - 1

    def create_arcs(
        self,
        k: int,
        distance_function: callable,
        pre_computed_distance: bool = False,
        pre_distances: Optional[np.array] = None,
    ) -> np.array:
        """Creates arcs for each node (adjacency relation).

        Args:
            k: Number of neighbours in the adjacency relation.
            distance_function: The distance function to be used to calculate the arcs.
            pre_computed_distance: Whether OPF should use a pre-computed distance or not.
            pre_distances: Pre-computed distance matrix.

        Returns:
            (np.array): The maximum possible distances for each value of k.

        """

        distances = np.zeros(k + 1)
        neighbours_idx = np.zeros(k + 1)
        max_distances = np.zeros(k)

        for i in range(self.n_nodes):
            distances.fill(c.FLOAT_MAX)

            for j in range(self.n_nodes):
                if j != i:
                    if pre_computed_distance:
                        distances[k] = pre_distances[self.nodes[i].idx][
                            self.nodes[j].idx
                        ]
                    else:
                        distances[k] = distance_function(
                            self.nodes[i].features, self.nodes[j].features
                        )

                    neighbours_idx[k] = j
                    cur_k = k

                    # While current `k` is bigger than 0 and the `k` distance is smaller than `k-1` distance
                    while cur_k > 0 and distances[cur_k] < distances[cur_k - 1]:
                        distances[cur_k], distances[cur_k - 1] = (
                            distances[cur_k - 1],
                            distances[cur_k],
                        )

                        neighbours_idx[cur_k], neighbours_idx[cur_k - 1] = (
                            neighbours_idx[cur_k - 1],
                            neighbours_idx[cur_k],
                        )

                        cur_k -= 1

            self.nodes[i].radius = 0.0
            self.nodes[i].n_plateaus = 0

            for l in range(k - 1, -1, -1):
                if distances[l] != c.FLOAT_MAX:
                    if distances[l] > self.density:
                        self.density = distances[l]
                    if distances[l] > self.nodes[i].radius:
                        self.nodes[i].radius = distances[l]
                    if distances[l] > max_distances[l]:
                        max_distances[l] = distances[l]

                    self.nodes[i].adjacency.insert(0, neighbours_idx[l])

        if self.density < 0.00001:
            self.density = 1

        return max_distances

    def eliminate_maxima_height(self, height: float) -> None:
        """Eliminates maxima values in the subgraph that are below the inputted height.

        Args:
            height: Height's threshold.

        """

        logger.debug("Eliminating maxima above height = %s ...", height)

        if height > 0:
            for i in range(self.n_nodes):
                self.nodes[i].cost = np.maximum(self.nodes[i].density - height, 0)

        logger.debug("Maxima eliminated.")
