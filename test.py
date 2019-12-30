import opfython.stream.loader as l
import opfython.stream.parser as p
from opfython.models.supervised import SupervisedOPF
from opfython.core.subgraph import Subgraph

# Loading a .txt file to a numpy array
txt = l.load_txt('data/sample.txt')

# Parsing a pre-loaded numpy array
X, Y = p.parse_array(txt)

# Creating a subgraph structure
g = Subgraph(X, Y)

#
opf = SupervisedOPF()

#
opf._find_prototypes(g)

for node in g.nodes:
    print(node.status)
