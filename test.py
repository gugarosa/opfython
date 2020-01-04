import opfython.stream.loader as l
import opfython.stream.parser as p
from opfython.core.subgraph import Subgraph
from opfython.models.supervised import SupervisedOPF

# Loading a .txt file to a numpy array
txt = l.load_txt('data/sample.txt')

# Parsing a pre-loaded numpy array
X, Y = p.parse_array(txt)

#
opf = SupervisedOPF()

#
opf.fit(X, Y)

opf.predict(X)
