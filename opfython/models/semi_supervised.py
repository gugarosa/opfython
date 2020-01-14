import time

import numpy as np

import opfython.utils.constants as c
import opfython.utils.exception as e
import opfython.utils.logging as l
from opfython.core.heap import Heap
from opfython.models.supervised import SupervisedOPF
from opfython.core.subgraph import Subgraph

logger = l.get_logger(__name__)


class SemiSupervisedOPF(SupervisedOPF):
    """A SemiSupervisedOPF which implements the semi-supervised version of OPF classifier.

    References:
        W. P. Amorim, A. X. Falcão and M. H. Carvalho. Semi-supervised Pattern Classification Using Optimum-Path Forest. 27th SIBGRAPI Conference on Graphics, Patterns and Images (2014).   

    """

    def __init__(self, distance='log_squared_euclidean', pre_computed_distance=None):
        """Initialization method.

        Args:
            distance (str): An indicator of the distance metric to be used.
            pre_computed_distance (str): A pre-computed distance file for feeding into OPF.

        """

        logger.info('Overriding class: SupervisedOPF -> SemiSupervisedOPF.')

        # Override its parent class with the receiving arguments
        super(SemiSupervisedOPF, self).__init__(
            distance=distance, pre_computed_distance=pre_computed_distance)

        logger.info('Class overrided.')