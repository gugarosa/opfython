import opfython.stream.parser as p
from opfython.stream import loader

# Loading a .txt file to a numpy array
txt = loader.load_txt("data/boat.txt")

# Parsing a pre-loaded numpy array
X, Y = p.parse_loader(txt)
