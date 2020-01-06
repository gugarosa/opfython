import numpy as np

import opfython.utils.constants as c
import opfython.utils.exception as e
import opfython.utils.logging as l

logger = l.get_logger(__name__)


class Node:
    """A Node class is used as the lowest structure level in the OPF workflow.

    """

    def __init__(self, idx=0, label=1, features=None):
        """Initialization method.

        Args:
            idx (int): The node's identifier.
            label (int): The node's label.
            features (np.array): An array of features.

        """

        # Initially, we need to set the node's index
        self.idx = idx

        # We also need to set its label (true label)
        self.label = label

        # Also its possible predicted label
        self.predicted_label = 1

        # Array of features
        self.features = features

        # Cost of the node
        self.cost = 0.0

        # Density of the node
        self.density = 0.0

        # Whether the node is a prototype or not
        self.status = c.STANDARD

        # Identifier to the predecessor node
        self.pred = 0

        # Whether the node is relevant or not
        self.relavant = c.IRRELEVANT

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
        """int: Node's label (true label).

        """

        return self._label

    @label.setter
    def label(self, label):
        if not isinstance(label, int):
            raise e.TypeError('`label` should be an integer')
        if label < 1:
            raise e.ValueError('`label` should be >= 1')

        self._label = label

    @property
    def predicted_label(self):
        """int: Node's predicted label.

        """

        return self._predicted_label

    @predicted_label.setter
    def predicted_label(self, predicted_label):
        if not isinstance(predicted_label, int):
            raise e.TypeError('`predicted_label` should be an integer')
        if predicted_label < 1:
            raise e.ValueError('`predicted_label` should be >= 1')

        self._predicted_label = predicted_label

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

    @property
    def cost(self):
        """float: Node's cost.

        """

        return self._cost

    @cost.setter
    def cost(self, cost):
        if not (isinstance(cost, float) or isinstance(cost, int)):
            raise e.TypeError('`cost` should be a float or integer')

        self._cost = cost

    @property
    def density(self):
        """float: Node's density.

        """

        return self._density

    @density.setter
    def density(self, density):
        if not (isinstance(density, float) or isinstance(density, int)):
            raise e.TypeError('`density` should be a float or integer')

        self._density = density

    @property
    def status(self):
        """int: Whether the node is a prototype or not.

        """

        return self._status

    @status.setter
    def status(self, status):
        if status not in [c.STANDARD, c.PROTOTYPE]:
            raise e.TypeError('`status` should be `STANDARD` or `PROTOTYPE`')

        self._status = status

    @property
    def pred(self):
        """int: Identifier to the predecessor node.

        """

        return self._pred

    @pred.setter
    def pred(self, pred):
        if not isinstance(pred, int):
            raise e.TypeError('`pred` should be an integer')
        if pred < c.NIL:
            raise e.ValueError(
                '`pred` should have a value larger than `NIL`, e.g., -1')

        self._pred = pred

    @property
    def relevant(self):
        """int: Whether the node is relevant or not.

        """

        return self._relevant

    @relevant.setter
    def relevant(self, relevant):
        if relevant not in [c.RELEVANT, c.IRRELEVANT]:
            raise e.TypeError(
                '`relevant` should be `RELEVANT` or `IRRELEVANT`')

        self._relevant = relevant
