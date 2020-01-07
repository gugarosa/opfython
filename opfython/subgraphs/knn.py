import numpy as np

import opfython.utils.exception as e
import opfython.utils.logging as l
from opfython.core.subgraph import Subgraph

logger = l.get_logger(__name__)


class KNNSubgraph(Subgraph):
    """A KNNSubgraph is used to implement a k-nearest neightbours subgraph.

    """

    def __init__(self, X=None, Y=None, from_file=None):
        """Initialization method.

        Args:
            X (np.array): Array of features.
            Y (np.array): Array of labels.
            from_file (bool): Whether Subgraph should be directly created from a file.

        """

        logger.info('Overriding class: Subgraph -> KNNSubgraph.')

        # Override its parent class with the receiving arguments
        super(KNNSubgraph, self).__init__(X=X, Y=Y, from_file=from_file)

        # Number of adjacent nodes (k-nearest neighbours)
        self.best_k = 0

        # Constant for computing the probability density function (p.d.f.)
        self.k = 0.0

        # Density of the subgraph
        self.density = 0.0

        # Minimum density of the subgraph
        self.min_density = 0.0

        # Maximum density of the subgraph
        self.max_density = 0.0

        logger.info('Class overrided.')

    def eliminate_maxima_height(self, height):
        """Eliminates maxima values in the subgraph that are below the inputted height.

        Args:
            height (float): Height's threshold.

        """

        logger.debug(f'Eliminating maxima above height = {height} ...')

        # Checks if height is bigger than zero
        if height > 0:
            # For every possible node
            for i in range(self.nodes):
                # Calculates its new cost
                self.nodes[i].cost = np.maximum(
                    self.nodes[i].density - height, 0)

        logger.debug('Maxima eliminated.')

    def eliminate_maxima_area(self, area):
        """Eliminates maxima values in the subgraph that are below the inputted area.

        Args:
            area (float): Area's threshold.

        """

        logger.debug(f'Eliminating maxima above area = {area} ...')

        logger.debug('Maxima eliminated.')

    def eliminate_maxima_volume(self, volume):
        """Eliminates maxima values in the subgraph that are below the inputted volume.

        Args:
            volume (float): Volume's threshold.

        """

        logger.debug(f'Eliminating maxima above volume = {volume} ...')

        logger.debug('Maxima eliminated.')