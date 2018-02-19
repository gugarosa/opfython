""" This is the dataset's structure and its basic functions module.
"""

import numpy as np

from .sample import Sample
from ..utils.exception import ArgumentException


class Dataset(object):
    """ A dataset class to hold multiple instances of samples

        # Arguments
            n_samples: number of samples
            n_classes: number of classes
            n_features: number of features

        # Properties
            n_samples: number of samples.
            n_classes: number of classes.
            n_features: number of features.
            samples: list of samples.
    """

    def __init__(self, **kwargs):
        # These properties should be set by the user via keyword arguments.
        allowed_kwargs = {'n_samples',
                          'n_classes',
                          'n_features'
                          }
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        # Define all class variables as 'None'
        self.n_samples = None
        self.n_classes = None
        self.n_features = None
        self.samples = None

        # Check if arguments are supplied
        if 'n_samples' not in kwargs:
            raise ArgumentException('n_samples')
        if 'n_classes' not in kwargs:
            raise ArgumentException('n_classes')
        if 'n_features' not in kwargs:
            raise ArgumentException('n_features')

        # Apply arguments to class variables
        self.n_samples = kwargs['n_samples']
        self.n_classes = kwargs['n_classes']
        self.n_features = kwargs['n_features']

        # Create the feature vector based on number of features
        self.samples = [Sample(n_features=self.n_features)
                        for _ in range(self.n_samples)]
