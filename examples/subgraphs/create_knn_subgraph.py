import opfython.stream.loader as l
import opfython.stream.parser as p
from opfython.subgraphs import KNNSubgraph

# Defining an input file
input_file = 'data/boat.txt'

# Loading a .txt file to a dataframe
txt = l.load_txt(input_file)

# Parsing a pre-loaded dataframe
X, Y = p.parse_loader(txt)

# Creating a knn-subgraph structure
g = KNNSubgraph(X, Y)

# KNNSubgraph can also be directly created from a file
g = KNNSubgraph(from_file=input_file)
