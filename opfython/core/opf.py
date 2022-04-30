"""Optimum-Path Forest standard definitions.
"""

import pickle
from typing import List, Optional

import numpy as np

import opfython.math.distance as d
import opfython.utils.exception as e
from opfython.core import Subgraph
from opfython.stream import loader
from opfython.utils import logging

logger = logging.get_logger(__name__)


class OPF:
    """A basic class to define all common OPF-related methods.

    References:
        J. P. Papa, A. X. Falcão and C. T. N. Suzuki.
        LibOPF: A library for the design of optimum-path forest classifiers (2015).

    """

    def __init__(
        self,
        distance: Optional[str] = "log_squared_euclidean",
        pre_computed_distance: Optional[str] = None,
    ) -> None:
        """Initialization method.

        Args:
            distance: An indicator of the distance metric to be used.
            pre_computed_distance: A pre-computed distance file for feeding into OPF.

        """

        logger.info("Creating class: OPF.")

        # Initializing an empty subgraph
        self.subgraph = None

        # An indicator of the distance metric to be used
        self.distance = distance

        # Gathers the distance function as a property
        self.distance_fn = d.DISTANCES[distance]

        # If OPF should use a pre-computed distance
        if pre_computed_distance:
            # Marks the boolean indicator as True
            self.pre_computed_distance = True

            # Apply the distances matrix
            self._read_distances(pre_computed_distance)

        else:
            # Marks the boolean indicator as False
            self.pre_computed_distance = False

            # Marks the pre-distances property as None
            self.pre_distances = None

        logger.debug(
            "Distance: %s | Pre-computed distance: %s.",
            self.distance,
            self.pre_computed_distance,
        )
        logger.info("Class created.")

    @property
    def subgraph(self) -> Subgraph:
        """Subgraph's instance."""

        return self._subgraph

    @subgraph.setter
    def subgraph(self, subgraph: Subgraph) -> None:
        if subgraph is not None:
            if not isinstance(subgraph, Subgraph):
                raise e.TypeError("`subgraph` should be a subgraph")

        self._subgraph = subgraph

    @property
    def distance(self) -> str:
        """Distance metric to be used."""

        return self._distance

    @distance.setter
    def distance(self, distance: str) -> None:
        if distance not in [
            "additive_symmetric",
            "average_euclidean",
            "bhattacharyya",
            "bray_curtis",
            "canberra",
            "chebyshev",
            "chi_squared",
            "chord",
            "clark",
            "cosine",
            "dice",
            "divergence",
            "euclidean",
            "gaussian",
            "gower",
            "hamming",
            "hassanat",
            "hellinger",
            "jaccard",
            "jeffreys",
            "jensen",
            "jensen_shannon",
            "k_divergence",
            "kulczynski",
            "kullback_leibler",
            "log_euclidean",
            "log_squared_euclidean",
            "lorentzian",
            "manhattan",
            "matusita",
            "max_symmetric",
            "mean_censored_euclidean",
            "min_symmetric",
            "neyman",
            "non_intersection",
            "pearson",
            "sangvi",
            "soergel",
            "squared",
            "squared_chord",
            "squared_euclidean",
            "statistic",
            "topsoe",
            "vicis_symmetric1",
            "vicis_symmetric2",
            "vicis_symmetric3",
            "vicis_wave_hedges",
        ]:
            raise e.TypeError(
                "`distance` should be `additive_symmetric`, `average_euclidean`, `bhattacharyya`, "
                "`bray_curtis`, `canberra`, `chebyshev`, `chi_squared`, `chord`, `clark`, `cosine`, "
                "`dice`, `divergence`, `euclidean`, `gaussian`, `gower`, `hamming`, `hassanat`, `hellinger`, "
                "`jaccard`, `jeffreys`, `jensen`, `jensen_shannon`, `k_divergence`, `kulczynski`, "
                "`kullback_leibler`, `log_euclidean`, `log_squared_euclidean`, `lorentzian`, `manhattan`, "
                "`matusita`, `max_symmetric`, `mean_censored_euclidean`, `min_symmetric`, `neyman`, "
                "`non_intersection`, `pearson`, `sangvi`, `soergel`, `squared`, `squared_chord`, "
                "`squared_euclidean`, `statistic`, `topsoe`, `vicis_symmetric1`, `vicis_symmetric2`, "
                "`vicis_symmetric3` or `vicis_wave_hedges`"
            )

        self._distance = distance

    @property
    def distance_fn(self) -> callable:
        """Distance function to be used."""

        return self._distance_fn

    @distance_fn.setter
    def distance_fn(self, distance_fn: callable) -> None:
        if not callable(distance_fn):
            raise e.TypeError("`distance_fn` should be a callable")

        self._distance_fn = distance_fn

    @property
    def pre_computed_distance(self) -> bool:
        """Whether OPF should use a pre-computed distance or not."""

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
        if pre_distances is not None:
            if not isinstance(pre_distances, np.ndarray):
                raise e.TypeError("`pre_distances` should be a numpy array")

        self._pre_distances = pre_distances

    def _read_distances(self, file_name: str) -> None:
        """Reads the distance between nodes from a pre-defined file.

        Args:
            file_name: File to be loaded.

        """

        logger.debug("Running private method: read_distances().")

        # Getting file extension
        extension = file_name.split(".")[-1]

        if extension == "csv":
            distances = loader.load_csv(file_name)

        elif extension == "txt":
            distances = loader.load_txt(file_name)

        else:
            # Raises an ArgumentError exception
            raise e.ArgumentError(
                "File extension not recognized. It should be either `.csv` or .txt`"
            )

        # Check if distances have been properly loaded
        if distances is None:
            raise e.ValueError("Pre-computed distances could not been properly loaded")

        # Apply the distances matrix to the property
        self.pre_distances = distances

    def load(self, file_name: str) -> None:
        """Loads the object from a pickle encoding.

        Args:
            file_name: Pickle's file path to be loaded.

        """

        logger.info("Loading model from file: %s ...", file_name)

        with open(file_name, "rb") as origin_file:
            opf = pickle.load(origin_file)

            self.__dict__.update(opf.__dict__)

        logger.info("Model loaded.")

    def save(self, file_name: str) -> None:
        """Saves the object to a pickle encoding.

        Args:
            file_name: File's name to be saved.

        """

        logger.info("Saving model to file: %s ...", file_name)

        with open(file_name, "wb") as dest_file:
            pickle.dump(self, dest_file)

        logger.info("Model saved.")

    def fit(self, X: np.array, Y: np.array) -> None:
        """Fits data in the classifier.

        It should be directly implemented in OPF child classes.

        Args:
            X: Array of features.
            Y: Array of labels.

        """

        raise NotImplementedError

    def predict(self, X: np.array) -> List[int]:
        """Predicts new data using the pre-trained classifier.

        It should be directly implemented in OPF child classes.

        Args:
            X: Array of features.

        Returns:
            (List[int]): A list of predictions for each record of the data.

        """

        raise NotImplementedError
