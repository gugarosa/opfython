import numpy as np
import opfython.utils.logging as l

logger = l.get_logger(__name__)


class Sample:
    """A sample class for all the input data (atomic level).

    Properties:
        label (int): Integer holding the label's identifier.
        n_features (int): Number of features.
        features (np.array): A numpy array to hold 'n' features.
        shape (tuple): A tuple containing the sample's shape.

    """

    def __init__(self, label=1, n_features=1, verbose=0):
        """Initialization method.

        Args:
            label (int): Integer holding the label's identifier.
            n_features (int): Number of features.
            verbose (int): Verbosity level.

        """

        logger.info('Creating class: Sample.')

        # Label of the sample
        self._label = label

        # Number of sample features
        self._n_features = n_features

        # Creating features array
        self._features = np.zeros(n_features)

        # Then, we calculate its shape
        self._shape = self._features.shape

        # We will log some important information
        if verbose:
            logger.debug(
                f'Label: {self._label} | Features: {self._features} | Shape: {self._shape}')

        logger.info('Class created.')

    @property
    def label(self):
        """An integer that holds the sample's label.
        """

        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def n_features(self):
        """The amount of features.
        """

        return self._features

    @property
    def features(self):
        """An array encoding sample's features.
        """

        return self._features

    @features.setter
    def features(self, features):
        self._features = features

    @property
    def shape(self):
        """A tuple that contains the sample's shape.
        """

        return self._shape
