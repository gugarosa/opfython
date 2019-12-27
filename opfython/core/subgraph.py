import opytimizer.utils.exception as e

import opfython.utils.logging as l
from opfython.core.node import Node

logger = l.get_logger(__name__)


class Subgraph():
    """A Subgraph class is used as a collection of Nodes and the basic structure to work with OPF.

    """

    def __init__(self, data):
        """Initialization method.

        Args:
            data (df): A pre-loaded dataframe in the OPF file format.

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

    @property
    def n_nodes(self):
        """int: Number of nodes.

        """

        return self._n_nodes

    @n_nodes.setter
    def n_nodes(self, n_nodes):
        if not isinstance(n_nodes, int):
            raise e.TypeError('`n_nodes` should be an integer')
        if n_nodes < 0:
            raise e.ValueError('`n_nodes` should be >= 0')

        self._n_nodes = n_nodes

    @property
    def n_features(self):
        """int: Number of features.

        """

        return self._n_features

    @n_features.setter
    def n_features(self, n_features):
        if not isinstance(n_features, int):
            raise e.TypeError('`n_features` should be an integer')
        if n_features < 0:
            raise e.ValueError('`n_features` should be >= 0')

        self._n_features = n_features

    @property
    def nodes(self):
        """list: List of Nodes that belongs to the Subgraph.

        """

        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        if not isinstance(nodes, list):
            raise e.TypeError('`nodes` should be a list')

        self._nodes = nodes

    @property
    def built(self):
        """bool: Indicate whether the function is built.

        """

        return self._built

    @built.setter
    def built(self, built):
        self._built = built

    def _build(self, data):
        """This method serves as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            data (df): A pre-loaded dataframe in the OPF file format.

        """

        logger.debug('Running private method: build().')

        # Iterate over every possible sample
        for idx, label, feature in zip(data['idx'], data['labels'], data['features']):
            # Creates a Node structure
            node = Node(idx, label, feature)

            # Appends the node to the list
            self.nodes.append(node)

        # Calculates the number of nodes
        self.n_nodes = len(self.nodes)

        # Calculates the number of features
        self.n_features = self.nodes[0].features.shape[0]

        # Set built variable to 'True'
        self.built = True

        # Logging attributes
        logger.debug(
            f'Nodes: {self.n_nodes} | Features: {self.n_features} | Built: {self.built}')
