import opfython.stream.parser as p
from opfython.core import Subgraph
from opfython.stream import loader

# Defining an input file
input_file = "data/boat.txt"

# Loading a .txt file to a dataframe
txt = loader.load_txt(input_file)

# Parsing a pre-loaded dataframe
X, Y = p.parse_loader(txt)

# Creating a subgraph structure
g = Subgraph(X, Y)

# Subgraph can also be directly created from a file
g = Subgraph(from_file=input_file)
