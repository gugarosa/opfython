import time

import numpy as np

import opfython.math.distance as d
import opfython.utils.constants as c
import opfython.utils.exception as e
import opfython.utils.logging as l
from opfython.core.heap import Heap
from opfython.core.opf import OPF
from opfython.core.subgraph import Subgraph

logger = l.get_logger(__name__)


class KNNSupervisedOPF(OPF):
    """A KNNSupervisedOPF which implements the supervised version of OPF classifier with a KNN subgraph.

    References:

    """

    def __init__(self, distance='log_squared_euclidean', pre_computed_distance=None):
        """Initialization method.

        Args:
            distance (str): An indicator of the distance metric to be used.
            pre_computed_distance (str): A pre-computed distance file for feeding into OPF.

        """

        logger.info('Overriding class: OPF -> KNNSupervisedOPF.')

        # Override its parent class with the receiving arguments
        super(KNNSupervisedOPF, self).__init__(
            distance=distance, pre_computed_distance=pre_computed_distance)

        logger.info('Class overrided.')