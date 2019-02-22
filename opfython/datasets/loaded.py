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

        """

        logger.info('Overriding class: Dataset -> Loaded.')

        # Getting file extension
        extension = file_path.split('.')[-1]

        if extension == 'txt':
            txt = self.load_txt(file_path)
        else:
            json = self.load_json(file_path)

        # Override its parent class with the receiving hyperparams
        # super(Loaded, self).__init__()

        logger.info('Class overrided.')

    def load_txt(self, txt_path):
        """
        """

        try:
            f = open(txt_path, "r")

            txt = f.read()

            return txt

        except FileNotFoundError:
            logger.error('File not found.')

    def load_json(self, json_path):
        """
        """

        json = None

        return json
