import numpy as np

import opfython.utils.logging as l

logger = l.get_logger(__name__)

class Sample:
    """ A sample class for all the input data (atomic level).

    Properties:
        label (int): Integer holding the label's identifier.
        features (np.array): A numpy array to hold 'n' features.

    """

    def __init__(self, label=1, features=None):
        """Initialization method.

        Args:
            label (int): Integer holding the label's identifier.
            features (np.array): A numpy array to hold 'n' features.

        """

        logger.info('Creating class: Sample.')

        # Initially, a sample needs its label and the features that represent it
        self._label = label
        self._features = features

        # Then, we calculate its shape
        self._shape = features.shape

         # We will log some important information
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
