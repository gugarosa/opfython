import opfython.math.general as g
import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s

# Loading a .txt file to a numpy array
txt = l.load_txt('data/boat.txt')

# Parsing a pre-loaded numpy array
X, Y = p.parse_loader(txt)

# Creating a file of pre-computed distances
g.pre_compute_distance(X, 'boat_split_distances.txt', distance='log_squared_euclidean')
