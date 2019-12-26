import numpy as np
import opfython.utils.logging as l

logger = l.get_logger(__name__)

class Node():


    def __init__(self, features, label=0):

        self.features = features
        self.label = label
