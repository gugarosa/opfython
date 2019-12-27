import opfython.utils.loader as l
import opfython.utils.parser as p
from opfython.core.opf import OPF
from opfython.core.subgraph import Subgraph

# Loading a .txt file to a dataframe
txt = l.load_txt('data/sample.txt')

# Parsing a pre-loaded dataframe
data = p.parse_df(txt)

# Creating a subgraph structure
s = Subgraph(data)

#
opf = OPF()

opf._find_prototypes(s)
