import opfython.stream.loader as l
import opfython.stream.parser as p
from opfython.core.opf import OPF
from opfython.core.subgraph import Subgraph

# Loading a .txt file to a numpy array
txt = l.load_txt('data/sample.txt')

# Parsing a pre-loaded numpy array
X, Y = p.parse_array(txt)

# Creating a subgraph structure
g = Subgraph(X, Y)

#
opf = OPF()

#
opf._find_prototypes(g)
