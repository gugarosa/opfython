import opfython.stream.loader as loader
import opfython.stream.parser as p
import opfython.utils.exception as e
import opfython.utils.logging as l
from opfython.core.node import Node

logger = l.get_logger(__name__)


class Subgraph:
    """A Subgraph class is used as a collection of Nodes and the basic structure to work with OPF.

    """

    def __init__(self, X=None, Y=None, from_file=None):
        """Initialization method.

        Args:
            X (np.array): Array of features.
            Y (np.array): Array of labels.
            from_file (bool): Whether Subgraph should be directly created from a file.

        """

        logger.debug('Creating class: Subgraph.')

        # Number of nodes
        self.n_nodes = 0

        # Number of features
        self.n_features = 0

        # List of nodes
        self.nodes = []

        # List of indexes of ordered nodes
        self.idx_nodes = []

        # Whether the subgraph is trained or not
        self.trained = False

        # Checks if data should be loaded from a file
        if from_file:
            # Loads the data
            X, Y = self._load(from_file)

        # Checks if data has been properly loaded
        if (X is not None) and (Y is not None):
            # Now, we need to build this class up
            self._build(X, Y)

            logger.debug('Class created.')

        # If data could not be loaded
        else:
            logger.error('Subgraph has not been properly created.')

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
    def idx_nodes(self):
        """list: List of indexes of ordered nodes.

        """

        return self._idx_nodes

    @idx_nodes.setter
    def idx_nodes(self, idx_nodes):
        if not isinstance(idx_nodes, list):
            raise e.TypeError('`idx_nodes` should be a list')

        self._idx_nodes = idx_nodes

    @property
    def trained(self):
        """bool: Indicate whether the subgraph is trained.

        """

        return self._trained

    @trained.setter
    def trained(self, trained):
        self._trained = trained

    def _load(self, file_path):
        """Loads and parses a dataframe from a file.

        Args:
            file_path (str): File to be loaded.

        Returns:
            Arrays holding the features and labels.

        """

        logger.debug('Running private method: load().')

        # Getting file extension
        extension = file_path.split('.')[-1]

        # Check if extension is .csv
        if extension == 'csv':
            # If yes, call the method that actually loads csv
            data = loader.load_csv(file_path)

        # Check if extension is .txt
        elif extension == 'txt':
            # If yes, call the method that actually loads txt
            data = loader.load_txt(file_path)

        # Check if extension is .json
        elif extension == 'json':
            # If yes, call the method that actually loads json
            data = loader.load_json(file_path)

        # If extension is not recognized
        else:
            # Raises a ValueError exception
            raise ValueError('File extension not recognized.')

        # Parsing array
        X, Y = p.parse_array(data)

        return X, Y

    def _build(self, X, Y):
        """This method serves as the object building process.

        One can define several commands here that does not necessarily
        needs to be on its initialization.

        Args:
            X (np.array): Features array.
            Y (np.array): Labels array.

        """

        logger.debug('Running private method: build().')

        # Iterate over every possible sample
        for i, (feature, label) in enumerate(zip(X, Y)):
            # Creates a Node structure
            node = Node(i, label.item(), feature)

            # Appends the node to the list
            self.nodes.append(node)

        # Calculates the number of nodes
        self.n_nodes = len(self.nodes)

        # Calculates the number of features
        self.n_features = self.nodes[0].features.shape[0]

        # Logging attributes
        logger.debug(
            f'Nodes: {self.n_nodes} | Features: {self.n_features} | Trained: {self.trained}.')
