import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models.unsupervised import UnsupervisedOPF

# Loading a .txt file to a numpy array
txt = l.load_txt('data/boat.txt')

# Parsing a pre-loaded numpy array
X, Y = p.parse_array(txt)

# Creates an UnsupervisedOPF instance
opf = UnsupervisedOPF(distance='log_squared_euclidean',
                    pre_computed_distance=None)

# Fits training data into the classifier
opf.fit(X)
