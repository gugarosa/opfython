import opfython.utils.logging as l
import opfython.utils.loader as loader
from opfython.core.dataset import Dataset
import numpy as np

logger = l.get_logger(__name__)


class Loaded(Dataset):
    """A dataset child that can load external data.

    Properties:

    Methods:

    """

    def __init__(self, file_path=None):
        """Initialization method.

        Args:
            file_path (str): A string holding the file's path (.txt or .json).

        """

        logger.info('Overriding class: Dataset -> Loaded.')

        # Getting file extension
        extension = file_path.split('.')[-1]

        # Check if extension is .txt
        if extension == 'csv':
            # If yes, call the method that actually loads txt
            data = loader.load_csv(file_path)
        
        #
        ids, labels, features = self._parse_data(data)

        n_samples = ids[-1]
        n_classes = np.argmax(labels)
        n_features = features[0].shape[0]

        print(n_samples, n_classes, n_features)

        # Override its parent class with the receiving hyperparams
        super(Loaded, self).__init__(n_samples=n_samples, n_classes=n_classes, n_features=n_features)

        #
        self.populate_samples(ids, labels, features)

        logger.info('Class overrided.')

    def _parse_data(self, data):
        """
        """

        ids = list(data[0])
        labels = list(data[1])
        features = list(data.iloc[:, 2:].values)

        return ids, labels, features
