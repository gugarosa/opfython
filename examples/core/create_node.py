import numpy as np

from opfython.core import Node

# Defining an index
idx = 0

# Defining a label
label = 1

# Defining an array of features
features = np.asarray([2, 2.5, 1.5, 4])

# Creating a Node
n = Node(idx, label, features)
