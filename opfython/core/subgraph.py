import numpy as np

import opfython.utils.logging as l
from opfython.core.node import Node

logger = l.get_logger(__name__)

class Subgraph():
    """
    """

    def __init__(self, data):
        """
        """

        logger.info('Creating class: Subgraph.')
        
        # Number of nodes
        self.n_nodes = 0

        # Number of features
        self.n_features = 0

        # List of nodes
        self.nodes = []

        # Whether the class is built or not
        self.built = False

        # Now, we need to build this class up
        self._build(data)

        logger.info('Class created.')


    def _build(self, data):
        """
        """

        logger.debug('Running private method: build().')
        
        #
        for idx, label, feature in zip(data['idx'], data['labels'], data['features']):
            #
            node = Node(idx, label, feature)

            #
            self.nodes.append(node)

        #
        self.n_nodes = len(self.nodes)

        #
        self.n_features = self.nodes[0].features.shape[0]
        
        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(f'Nodes: {self.n_nodes} | Features: {self.n_features} | Built: {self.built}')
