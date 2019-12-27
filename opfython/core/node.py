import numpy as np
import opfython.utils.logging as l

logger = l.get_logger(__name__)

class Node():
    """
    """


    def __init__(self, idx=0, label=0, features=None):
        """
        """

        #
        self.idx = idx

        #
        self.features = features

        #
        self.label = label
