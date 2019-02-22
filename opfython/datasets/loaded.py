import numpy as np

import opfython.utils.loader as loader
import opfython.utils.logging as l
from opfython.core.dataset import Dataset

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

        # Check if extension is .csv
        if extension == 'csv':
            # If yes, call the method that actually loads csv
            data = loader.load_csv(file_path)
        # Check if extension is .txt
        elif extension == 'txt':
            # If yes, call the method that actually loads txt
            data = loader.load_csv(file_path)
        # Check if extension is .json
        elif extension == 'json':
            # If yes, call the method that actually loads json
            data = loader.load_json(file_path)

        # Parsing dataframe
        ids, labels, features = loader.parse_df(data)

        # From id's, we can get the numbner of samples
        n_samples = ids[-1]

        # The maximum number of classes equal the biggest label
        n_classes = np.argmax(labels)

        # Also the first shape of features array holds its amount
        n_features = features[0].shape[0]

        # Override its parent class with the receiving hyperparams
        super(Loaded, self).__init__(n_samples=n_samples,
                                     n_classes=n_classes, n_features=n_features)

        # Populating samples from pre-loaded data
        self.populate_samples(labels, features)

        logger.info('Class overrided.')
