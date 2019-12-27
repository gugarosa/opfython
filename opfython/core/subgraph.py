import numpy as np
import opfython.utils.logging as l

logger = l.get_logger(__name__)

class Subgraph():

    def __init__(self, dataset):

        self.n_nodes = 1
        self.n_features = 1
        self.nodes = []

