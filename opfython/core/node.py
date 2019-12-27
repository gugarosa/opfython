import numpy as np
import opytimizer.utils.exception as e

import opfython.utils.logging as l

logger = l.get_logger(__name__)


class Node:
    """A Node class is used as the lowest structure level in the OPF workflow.

    """

    def __init__(self, idx=0, label=0, features=None):
        """Initialization method.

        Args:
            idx (int): The node's identifier.
            label (int): The node's label.
            features (np.array): An array of features.

        """

        # Initially, we need to set the node's index
        self.idx = idx

        # We also need to set its label
        self.label = label

        # And finally, its array of features
        self.features = features

        self.status = 0

        self.pred = 0

        self.cost = 0.0

        self.true_label = 0

    @property
    def idx(self):
        """int: Node's index.

        """

        return self._idx

    @idx.setter
    def idx(self, idx):
        if not isinstance(idx, int):
            raise e.TypeError('`idx` should be an integer')
        if idx < 0:
            raise e.ValueError('`idx` should be >= 0')

        self._idx = idx

    @property
    def label(self):
        """int: Node's label.

        """

        return self._label

    @label.setter
    def label(self, label):
        if not isinstance(label, int):
            raise e.TypeError('`label` should be an integer')
        if label < 0:
            raise e.ValueError('`label` should be >= 0')

        self._label = label

    @property
    def features(self):
        """np.array: N-dimensional array of features.

        """

        return self._features

    @features.setter
    def features(self, features):
        if not isinstance(features, np.ndarray):
            raise e.TypeError('`features` should be a numpy array')

        self._features = features
