# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Represent the mutable samples used by Optimum-Path Forest models."""

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike

import opfython.utils.constants as c
import opfython.utils.exception as e
from opfython.utils.logging import get_logger

logger = get_logger(__name__)

_Number: TypeAlias = float | int | np.int32 | np.int64


class Node:
    """Hold one sample and its mutable forest state."""

    def __init__(
        self,
        idx: int = 0,
        label: int = 0,
        features: ArrayLike | None = None,
    ) -> None:
        """Initialize a sample with unassigned forest state.

        Feature conversion can retain the caller's array storage. A node without
        features is a placeholder and cannot participate in distance calculations.

        Args:
            idx: Nonnegative sample index used to address precomputed distances.
            label: Nonnegative integer class label.
            features: Sample features converted with NumPy without forcing a copy.

        Raises:
            opfython.utils.exception.TypeError: The index or label is not a Python integer.
            opfython.utils.exception.ValueError: The index or label is negative.

        """

        self.idx = idx

        self.label = label
        self.predicted_label = 0
        self.cluster_label = 0

        self.features = np.asarray(features)

        self.cost = 0.0
        self.density = 0.0
        self.radius = 0.0

        self.n_plateaus = 0
        self.adjacency = []
        self.root = 0

        self.status = c.STANDARD
        self.pred = c.NIL
        self.relevant = c.IRRELEVANT

    @property
    def idx(self) -> int:
        """Return the original sample index."""

        return self._idx

    @idx.setter
    def idx(self, idx: int) -> None:
        if not isinstance(idx, int):
            raise e.TypeError(f"`idx` should be an integer, but got {type(idx).__name__}.")
        if idx < 0:
            raise e.ValueError(f"`idx` should be >= 0, but got {idx}.")

        self._idx = idx

    @property
    def label(self) -> int:
        """Return the sample's true class label."""

        return self._label

    @label.setter
    def label(self, label: int) -> None:
        if not isinstance(label, int):
            raise e.TypeError(f"`label` should be an integer, but got {type(label).__name__}.")
        if label < 0:
            raise e.ValueError(f"`label` should be >= 0, but got {label}.")

        self._label = label

    @property
    def predicted_label(self) -> int:
        """Return the class label assigned by the forest."""

        return self._predicted_label

    @predicted_label.setter
    def predicted_label(self, predicted_label: int) -> None:
        if not isinstance(predicted_label, int):
            raise e.TypeError(f"`predicted_label` should be an integer, but got {type(predicted_label).__name__}.")
        if predicted_label < 0:
            raise e.ValueError(f"`predicted_label` should be >= 0, but got {predicted_label}.")

        self._predicted_label = predicted_label

    @property
    def cluster_label(self) -> int:
        """Return the cluster assigned by an unsupervised forest."""

        return self._cluster_label

    @cluster_label.setter
    def cluster_label(self, cluster_label: int) -> None:
        if not isinstance(cluster_label, int):
            raise e.TypeError(f"`cluster_label` should be an integer, but got {type(cluster_label).__name__}.")
        if cluster_label < 0:
            raise e.ValueError(f"`cluster_label` should be >= 0, but got {cluster_label}.")

        self._cluster_label = cluster_label

    @property
    def features(self) -> np.ndarray:
        """Return the feature storage retained by this node."""

        return self._features

    @features.setter
    def features(self, features: np.ndarray) -> None:
        if not isinstance(features, np.ndarray):
            raise e.TypeError(f"`features` should be a NumPy array, but got {type(features).__name__}.")

        self._features = features

    @property
    def cost(self) -> _Number:
        """Return the path cost assigned by the forest."""

        return self._cost

    @cost.setter
    def cost(self, cost: _Number) -> None:
        if not isinstance(cost, _Number):
            raise e.TypeError(f"`cost` should be a float or supported integer, but got {type(cost).__name__}.")

        self._cost = cost

    @property
    def density(self) -> _Number:
        """Return the scaled probability density at this sample."""

        return self._density

    @density.setter
    def density(self, density: _Number) -> None:
        if not isinstance(density, _Number):
            raise e.TypeError(f"`density` should be a float or supported integer, but got {type(density).__name__}.")

        self._density = density

    @property
    def radius(self) -> _Number:
        """Return the maximum distance to a selected neighbour."""

        return self._radius

    @radius.setter
    def radius(self, radius: _Number) -> None:
        if not isinstance(radius, _Number):
            raise e.TypeError(f"`radius` should be a float or supported integer, but got {type(radius).__name__}.")

        self._radius = radius

    @property
    def n_plateaus(self) -> int:
        """Return the number of added plateau neighbours."""

        return self._n_plateaus

    @n_plateaus.setter
    def n_plateaus(self, n_plateaus: int) -> None:
        if not isinstance(n_plateaus, int):
            raise e.TypeError(f"`n_plateaus` should be an integer, but got {type(n_plateaus).__name__}.")
        if n_plateaus < 0:
            raise e.ValueError(f"`n_plateaus` should be >= 0, but got {n_plateaus}.")

        self._n_plateaus = n_plateaus

    @property
    def adjacency(self) -> list[int | float]:
        """Return adjacent graph positions stored as integers or integer-valued floats."""

        return self._adjacency

    @adjacency.setter
    def adjacency(self, adjacency: list[int | float]) -> None:
        if not isinstance(adjacency, list):
            raise e.TypeError(f"`adjacency` should be a list, but got {type(adjacency).__name__}.")

        self._adjacency = adjacency

    @property
    def root(self) -> int:
        """Return the root's position in the subgraph."""

        return self._root

    @root.setter
    def root(self, root: int) -> None:
        if not isinstance(root, int):
            raise e.TypeError(f"`root` should be an integer, but got {type(root).__name__}.")
        if root < 0:
            raise e.ValueError(f"`root` should be >= 0, but got {root}.")

        self._root = root

    @property
    def status(self) -> int:
        """Return whether the sample is a prototype."""

        return self._status

    @status.setter
    def status(self, status: int) -> None:
        if status not in [c.STANDARD, c.PROTOTYPE]:
            raise e.TypeError(f"`status` should be `STANDARD` or `PROTOTYPE`, but got {status}.")

        self._status = status

    @property
    def pred(self) -> int:
        """Return the predecessor's graph position or the NIL sentinel."""

        return self._pred

    @pred.setter
    def pred(self, pred: int) -> None:
        if not isinstance(pred, int):
            raise e.TypeError(f"`pred` should be an integer, but got {type(pred).__name__}.")
        if pred < c.NIL:
            raise e.ValueError(f"`pred` should be >= `NIL`, but got {pred}.")

        self._pred = pred

    @property
    def relevant(self) -> int:
        """Return whether the sample belongs to a relevant prediction path."""

        return self._relevant

    @relevant.setter
    def relevant(self, relevant: int) -> None:
        if relevant not in [c.RELEVANT, c.IRRELEVANT]:
            raise e.TypeError(f"`relevant` should be `RELEVANT` or `IRRELEVANT`, but got {relevant}.")

        self._relevant = relevant
