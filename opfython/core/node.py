"""Node structure that belongs to the Optimum-Path Forest.
"""

from typing import List, Optional

import numpy as np

import opfython.utils.constants as c
import opfython.utils.exception as e
from opfython.utils import logging

logger = logging.get_logger(__name__)


class Node:
    """A Node class is used as the lowest structure level in the OPF workflow."""

    def __init__(
        self,
        idx: int = 0,
        label: int = 0,
        features: Optional[np.array] = None,
    ) -> None:
        """Initialization method.

        Args:
            idx: The node's identifier.
            label: The node's label.
            features: An array of features.

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
        """Node's index."""

        return self._idx

    @idx.setter
    def idx(self, idx: int) -> None:
        if not isinstance(idx, int):
            raise e.TypeError("`idx` should be an integer")
        if idx < 0:
            raise e.ValueError("`idx` should be >= 0")

        self._idx = idx

    @property
    def label(self) -> int:
        """Node's label (true label)."""

        return self._label

    @label.setter
    def label(self, label: int) -> None:
        if not isinstance(label, int):
            raise e.TypeError("`label` should be an integer")
        if label < 0:
            raise e.ValueError("`label` should be >= 0")

        self._label = label

    @property
    def predicted_label(self) -> int:
        """Node's predicted label."""

        return self._predicted_label

    @predicted_label.setter
    def predicted_label(self, predicted_label: int) -> None:
        if not isinstance(predicted_label, int):
            raise e.TypeError("`predicted_label` should be an integer")
        if predicted_label < 0:
            raise e.ValueError("`predicted_label` should be >= 0")

        self._predicted_label = predicted_label

    @property
    def cluster_label(self) -> int:
        """Node's cluster assignment identifier."""

        return self._cluster_label

    @cluster_label.setter
    def cluster_label(self, cluster_label: int) -> None:
        if not isinstance(cluster_label, int):
            raise e.TypeError("`cluster_label` should be an integer")
        if cluster_label < 0:
            raise e.ValueError("`cluster_label` should be >= 0")

        self._cluster_label = cluster_label

    @property
    def features(self) -> np.array:
        """np.array: N-dimensional array of features."""

        return self._features

    @features.setter
    def features(self, features: np.array) -> None:
        if not isinstance(features, np.ndarray):
            raise e.TypeError("`features` should be a numpy array")

        self._features = features

    @property
    def cost(self) -> float:
        """Node's cost."""

        return self._cost

    @cost.setter
    def cost(self, cost: float) -> None:
        if not isinstance(cost, (float, int, np.int32, np.int64)):
            raise e.TypeError("`cost` should be a float or integer")

        self._cost = cost

    @property
    def density(self) -> float:
        """Node's density."""

        return self._density

    @density.setter
    def density(self, density: float) -> None:
        if not isinstance(density, (float, int, np.int32, np.int64)):
            raise e.TypeError("`density` should be a float or integer")

        self._density = density

    @property
    def radius(self) -> float:
        """Maximum distance among the k-nearest neighbors."""

        return self._radius

    @radius.setter
    def radius(self, radius: float) -> None:
        if not isinstance(radius, (float, int, np.int32, np.int64)):
            raise e.TypeError("`radius` should be a float or integer")

        self._radius = radius

    @property
    def n_plateaus(self) -> int:
        """Amount of adjacent nodes on plateaus."""

        return self._n_plateaus

    @n_plateaus.setter
    def n_plateaus(self, n_plateaus: int) -> None:
        if not isinstance(n_plateaus, int):
            raise e.TypeError("`n_plateaus` should be an integer")
        if n_plateaus < 0:
            raise e.ValueError("`n_plateaus` should be >= 0")

        self._n_plateaus = n_plateaus

    @property
    def adjacency(self) -> List[int]:
        """Adjacent nodes."""

        return self._adjacency

    @adjacency.setter
    def adjacency(self, adjacency: List[int]) -> None:
        if not isinstance(adjacency, list):
            raise e.TypeError("`adjacency` should be a list")

        self._adjacency = adjacency

    @property
    def root(self) -> int:
        """Cluster's root node identifier."""

        return self._root

    @root.setter
    def root(self, root: int) -> None:
        if not isinstance(root, int):
            raise e.TypeError("`root` should be an integer")
        if root < 0:
            raise e.ValueError("`root` should be >= 0")

        self._root = root

    @property
    def status(self) -> int:
        """Whether the node is a prototype or not."""

        return self._status

    @status.setter
    def status(self, status: int) -> None:
        if status not in [c.STANDARD, c.PROTOTYPE]:
            raise e.TypeError("`status` should be `STANDARD` or `PROTOTYPE`")

        self._status = status

    @property
    def pred(self) -> int:
        """Identifier to the predecessor node."""

        return self._pred

    @pred.setter
    def pred(self, pred: int) -> None:
        if not isinstance(pred, int):
            raise e.TypeError("`pred` should be an integer")
        if pred < c.NIL:
            raise e.ValueError("`pred` should have a value larger than `NIL`, e.g., -1")

        self._pred = pred

    @property
    def relevant(self) -> int:
        """Whether the node is relevant or not."""

        return self._relevant

    @relevant.setter
    def relevant(self, relevant: int) -> None:
        if relevant not in [c.RELEVANT, c.IRRELEVANT]:
            raise e.TypeError("`relevant` should be `RELEVANT` or `IRRELEVANT`")

        self._relevant = relevant
