import opfython.utils.loader as l
import opfython.utils.parser as p
from opfython.core.subgraph import Subgraph

# Defining an input file
input_file = 'data/samplel.txt'

# Loading a .txt file to a dataframe
txt = l.load_txt(input_file)

# Parsing a pre-loaded dataframe
data = p.parse_df(txt)

# Creating a subgraph structure
s = Subgraph(data)

# Subgraph can also be directly created from a file
s = Subgraph(from_file=input_file)
