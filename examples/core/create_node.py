import numpy as np

from opfython.core import Node

# Defining index, label and features
idx = 0
label = 0
features = np.asarray([2, 2.5, 1.5, 4])

# Creating a Node
n = Node(idx, label, features)
