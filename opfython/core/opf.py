"""Optimum-Path Forest standard definitions."""

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np

import opfython.math.distance as distance_module
import opfython.utils.exception as e
from opfython.core.subgraph import Subgraph
from opfython.stream import loader
from opfython.utils import logging

logger = logging.get_logger(__name__)


class OPF:
    """Common functionality for OPF classifiers."""

    def __init__(
        self,
        distance: str = "log_squared_euclidean",
        pre_computed_distance: Optional[str] = None,
    ) -> None:
        logger.info("Creating class: OPF.")

        self.subgraph = None
        self.distance = distance
        self.distance_fn = distance_module.DISTANCES[distance]

        if pre_computed_distance:
            self.pre_computed_distance = True
            self._read_distances(pre_computed_distance)
        else:
            self.pre_computed_distance = False
            self.pre_distances = None

        logger.debug(
            "Distance: %s | Pre-computed distance: %s.",
            self.distance,
            self.pre_computed_distance,
        )
        logger.info("Class created.")

    @property
    def subgraph(self) -> Subgraph:
        """Classifier subgraph."""

        return self._subgraph

    @subgraph.setter
    def subgraph(self, subgraph: Subgraph) -> None:
        if subgraph is not None and not isinstance(subgraph, Subgraph):
            raise e.TypeError("`subgraph` should be a subgraph")
        self._subgraph = subgraph

    @property
    def distance(self) -> str:
        """Distance metric name."""

        return self._distance

    @distance.setter
    def distance(self, distance: str) -> None:
        if not isinstance(distance, str):
            raise e.TypeError("`distance` should be a string")
        if distance not in distance_module.DISTANCES:
            raise e.TypeError(f"Unknown distance metric: {distance}")
        self._distance = distance

    @property
    def distance_fn(self) -> callable:
        """Distance callable."""

        return self._distance_fn

    @distance_fn.setter
    def distance_fn(self, distance_fn: callable) -> None:
        if not callable(distance_fn):
            raise e.TypeError("`distance_fn` should be a callable")
        self._distance_fn = distance_fn

    @property
    def pre_computed_distance(self) -> bool:
        """Whether to use pre-computed distances."""

        return self._pre_computed_distance

    @pre_computed_distance.setter
    def pre_computed_distance(self, pre_computed_distance: bool) -> None:
        if not isinstance(pre_computed_distance, bool):
            raise e.TypeError("`pre_computed_distance` should be a boolean")
        self._pre_computed_distance = pre_computed_distance

    @property
    def pre_distances(self) -> np.array:
        """Pre-computed distance matrix."""

        return self._pre_distances

    @pre_distances.setter
    def pre_distances(self, pre_distances: np.array) -> None:
        if pre_distances is not None and not isinstance(pre_distances, np.ndarray):
            raise e.TypeError("`pre_distances` should be a numpy array")
        self._pre_distances = pre_distances

    def _read_distances(self, file_name: str) -> None:
        """Read a pre-computed distance matrix."""

        logger.debug("Running private method: read_distances().")

        suffix = Path(file_name).suffix.lower()
        loaders = {".csv": loader.load_csv, ".txt": loader.load_txt}
        try:
            load = loaders[suffix]
        except KeyError as error:
            raise e.ArgumentError(
                "File extension should be either `.csv` or `.txt`"
            ) from error

        distances = load(file_name)
        if distances is None:
            raise e.ValueError("Pre-computed distances could not be loaded")

        self.pre_distances = distances

    def get_distances(self, normalize: bool = False) -> np.array:
        """Return the pairwise distance matrix for the current subgraph."""

        distances = np.zeros((self.subgraph.n_nodes, self.subgraph.n_nodes))

        for i in range(self.subgraph.n_nodes):
            for j in range(self.subgraph.n_nodes):
                distances[i, j] = self.distance_fn(
                    self.subgraph.nodes[i].features,
                    self.subgraph.nodes[j].features,
                )

        if normalize:
            return (distances - distances.min()) / (distances.max() - distances.min())

        return distances

    def load(self, file_name: str) -> None:
        """Load a serialized classifier."""

        logger.info("Loading model from file: %s ...", file_name)
        with open(file_name, "rb") as origin_file:
            self.__dict__.update(pickle.load(origin_file).__dict__)
        logger.info("Model loaded.")

    def save(self, file_name: str) -> None:
        """Serialize the classifier."""

        logger.info("Saving model to file: %s ...", file_name)
        with open(file_name, "wb") as destination_file:
            pickle.dump(self, destination_file)
        logger.info("Model saved.")

    def fit(self, X: np.array, Y: np.array) -> None:
        """Fit data in a concrete classifier."""

        raise NotImplementedError

    def predict(self, X: np.array) -> List[int]:
        """Predict data with a concrete classifier."""

        raise NotImplementedError
