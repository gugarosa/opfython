import opfython.utils.exception as e
import opfython.stream.loader as loader
import opfython.utils.logging as l
import opfython.stream.parser as p
from opfython.core.node import Node

logger = l.get_logger(__name__)


class Subgraph:
    """A Subgraph class is used as a collection of Nodes and the basic structure to work with OPF.

    """

    def __init__(self, data=None, from_file=None):
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

        # Checks if data should be loaded from a file
        if from_file:
            # Loads the data
            data = self._load(from_file)

        # Checks if data has been properly loaded
        if data:
            # Now, we need to build this class up
            self._build(data)

            logger.info('Class created.')

        # If data could not be loaded
        else:
            logger.warn('Subgraph has not been properly created.')

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

    def _load(self, file_path):
        """Loads and parses a dataframe from a file.

        Args:
            file_path (str): File to be loaded.

        Returns:
            A parsed data dictionary.

        """

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

        # Parsing dataframe
        data = p.parse_df(data)

        return data

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
