# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Provide configuration and persistence shared by OPF models."""

import pickle
from collections.abc import Callable
from os import PathLike
from pathlib import Path

import numpy as np

import opfython.math.distance as distance_module
import opfython.utils.exception as e
from opfython.core.subgraph import Subgraph
from opfython.stream import loader
from opfython.utils.logging import get_logger

logger = get_logger(__name__)


class OPF:
    """Provide common configuration, distance, and persistence operations."""

    def __init__(
        self,
        distance: str = "log_squared_euclidean",
        pre_computed_distance: str | PathLike[str] | None = None,
    ) -> None:
        """Initialize the metric and optional precomputed distance storage.

        Concrete models supply training and prediction. The base object also
        supports distance generation after a subgraph has been assigned.

        Args:
            distance: Metric name registered in opfython.math.distance.DISTANCES.
            pre_computed_distance: CSV or text file containing indexed pairwise distances.

        Raises:
            opfython.utils.exception.TypeError: The metric name is invalid.
            opfython.utils.exception.ArgumentError: The distance file extension is unsupported.
            opfython.utils.exception.ValueError: The distance file cannot be loaded.

        References:
            J. P. Papa, A. X. Falcao and C. T. N. Suzuki.
            LibOPF: A library for the design of optimum-path forest classifiers (2015).

        """

        logger.info("Creating class: OPF.")

        self.subgraph = None
        self.distance = distance

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
    def subgraph(self) -> Subgraph | None:
        """Return the current graph or None before a graph is assigned."""

        return self._subgraph

    @subgraph.setter
    def subgraph(self, subgraph: Subgraph | None) -> None:
        if subgraph is not None and not isinstance(subgraph, Subgraph):
            raise e.TypeError(f"`subgraph` should be a Subgraph or None, but got {type(subgraph).__name__}.")

        self._subgraph = subgraph

    @property
    def distance(self) -> str:
        """Return the registered metric name selected for distance calculation.

        Setting this name also selects its registered callable. Refit a trained
        classifier after changing its metric configuration.

        """

        return self._distance

    @distance.setter
    def distance(self, distance: str) -> None:
        if not isinstance(distance, str):
            raise e.TypeError(f"`distance` should be a string, but got {type(distance).__name__}.")
        if distance not in distance_module.DISTANCES:
            raise e.TypeError(f"`distance` should name a registered metric, but got {distance!r}.")

        self.distance_fn = distance_module.DISTANCES[distance]
        self._distance = distance

    @property
    def distance_fn(self) -> Callable[[np.ndarray, np.ndarray], float]:
        """Return the callable used for live distance calculations.

        A custom callable receives two feature arrays and returns a scalar
        distance. Selecting a registered metric name replaces this override.

        """

        return self._distance_fn

    @distance_fn.setter
    def distance_fn(self, distance_fn: Callable[[np.ndarray, np.ndarray], float]) -> None:
        if not callable(distance_fn):
            raise e.TypeError(f"`distance_fn` should be callable, but got {type(distance_fn).__name__}.")

        self._distance_fn = distance_fn

    @property
    def pre_computed_distance(self) -> bool:
        """Return whether fitting and prediction use the stored distance matrix."""

        return self._pre_computed_distance

    @pre_computed_distance.setter
    def pre_computed_distance(self, pre_computed_distance: bool) -> None:
        if not isinstance(pre_computed_distance, bool):
            raise e.TypeError(
                f"`pre_computed_distance` should be a boolean, but got {type(pre_computed_distance).__name__}."
            )

        self._pre_computed_distance = pre_computed_distance

    @property
    def pre_distances(self) -> np.ndarray | None:
        """Return the stored distance matrix or None when no matrix is loaded."""

        return self._pre_distances

    @pre_distances.setter
    def pre_distances(self, pre_distances: np.ndarray | None) -> None:
        if pre_distances is not None and not isinstance(pre_distances, np.ndarray):
            raise e.TypeError(
                f"`pre_distances` should be a NumPy array or None, but got {type(pre_distances).__name__}."
            )

        self._pre_distances = pre_distances

    def _read_distances(self, file_name: str | PathLike[str]) -> None:
        logger.debug("Running private method: read_distances().")

        suffix = Path(file_name).suffix.lower()
        loaders = {".csv": loader.load_csv, ".txt": loader.load_txt}

        try:
            load = loaders[suffix]
        except KeyError as error:
            raise e.ArgumentError(f"`file_name` should end with .csv or .txt, but got {suffix!r}.") from error

        distances = load(file_name)
        if distances is None:
            raise e.ValueError(f"`file_name` could not be loaded, but got {file_name!r}.")

        self.pre_distances = distances

    def _validate_pre_distances(self, rows: Subgraph, columns: Subgraph | None = None) -> None:
        if not self.pre_computed_distance:
            return

        if self.pre_distances is None or self.pre_distances.ndim != 2:
            raise e.BuildError("`pre_distances` should be a two-dimensional array.")

        if columns is None:
            columns = rows

        for graph, size in zip((rows, columns), self.pre_distances.shape):
            if any(node.idx >= size for node in graph.nodes):
                raise e.BuildError("`pre_distances` should cover every sample index on the accessed axis.")

    def get_distances(self, normalize: bool = False) -> np.ndarray:
        """Calculate pairwise distances for the current subgraph.

        The operation does not require a fitted model. Normalization follows
        NumPy division semantics when the distance range is zero.

        Args:
            normalize: Whether to apply min-max scaling to the calculated matrix.

        Returns:
            A floating-point array with shape (n_nodes, n_nodes).

        Raises:
            opfython.utils.exception.BuildError: No subgraph is assigned.

        """

        subgraph = self.subgraph
        if subgraph is None:
            raise e.BuildError("`subgraph` is None.")

        distances = np.zeros((subgraph.n_nodes, subgraph.n_nodes))

        for i in range(subgraph.n_nodes):
            for j in range(subgraph.n_nodes):
                distances[i, j] = self.distance_fn(
                    subgraph.nodes[i].features,
                    subgraph.nodes[j].features,
                )

        if normalize:
            return (distances - distances.min()) / (distances.max() - distances.min())

        return distances

    def load(self, file_name: str | PathLike[str]) -> None:
        """Restore a trusted serialized classifier into this instance.

        Use an instance of the saved model's class. Loading mutates its state
        in place, and I/O or deserialization failures propagate. Pickle can
        execute code and must not be loaded from untrusted sources.

        Args:
            file_name: Path to the trusted model pickle.

        Raises:
            OSError: The model file cannot be read.

        """

        logger.info("Loading model from file: %s ...", file_name)

        with open(file_name, "rb") as origin_file:
            self.__dict__.update(pickle.load(origin_file).__dict__)

        logger.info("Model loaded.")

    def save(self, file_name: str | PathLike[str]) -> None:
        """Serialize the current model state as a pickle.

        The destination is overwritten. I/O and serialization failures
        propagate, and compatible model code and dependencies are needed to load it.

        Args:
            file_name: Destination for the serialized model.

        Raises:
            OSError: The destination cannot be written.

        """

        logger.info("Saving model to file: %s ...", file_name)

        with open(file_name, "wb") as destination_file:
            pickle.dump(self, destination_file)

        logger.info("Model saved.")

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Define the training operation supplied by concrete classifiers.

        Args:
            X: Training feature array.
            Y: Training label array.

        Raises:
            NotImplementedError: The base class does not implement training.

        """

        raise NotImplementedError("`fit` should be implemented by a concrete classifier.")

    def predict(self, X: np.ndarray) -> list[int]:
        """Define the prediction operation supplied by concrete classifiers.

        Args:
            X: Feature array to predict.

        Raises:
            NotImplementedError: The base class does not implement prediction.

        """

        raise NotImplementedError("`predict` should be implemented by a concrete classifier.")
