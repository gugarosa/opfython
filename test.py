import opfython.stream.loader as l
import opfython.stream.parser as p
from opfython.core.opf import OPF
from opfython.core.subgraph import Subgraph

import numpy as np
import opfython.math.distance as d

# # Loading a .txt file to a dataframe
# txt = l.load_txt('data/sample.txt')

# # Parsing a pre-loaded dataframe
# data = p.parse_df(txt)

# # Creating a subgraph structure
# s = Subgraph(data)

# #
# opf = OPF()

# opf._find_prototypes(s)

x = np.asarray([2, 3, 4, 5])
y = np.asarray([1, 2, 3, 1])

dist = d.squared_cord_distance(x, y)
print(dist)
