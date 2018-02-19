""" This is the sample's structure and its basic functions module.
"""

import numpy as np

from ..utils.exception import ArgumentException


class Sample(object):
    """ A Sample class for all the input data.

        # Arguments
            n_features: number of features.

        # Properties
            label: integer identifier of sample's label.
            features: [n_features] vector to hold features' values.
    """

    def __init__(self, **kwargs):
        # These properties should be set by the user via keyword arguments.
        allowed_kwargs = {'n_features'
                          }
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        # Define all class variables as 'None'
        self.label = None
        self.features = None

        # Check if arguments are supplied
        if 'n_features' not in kwargs:
            raise ArgumentException('n_features')

        # Apply arguments to class variables
        __n_features = kwargs['n_features']

        # Instanciate the class label as -1
        self.label = -1

        # Create the feature vector based on number of features
        self.features = np.zeros(__n_features)
