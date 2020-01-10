import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models.unsupervised import UnsupervisedOPF

# Loading a .txt file to a numpy array
txt = l.load_txt('data/boat.txt')

# Parsing a pre-loaded numpy array
X, Y = p.parse_array(txt)

# Creates an UnsupervisedOPF instance
opf = UnsupervisedOPF(min_k=1, max_k=10, distance='log_squared_euclidean', pre_computed_distance=None)

# Fits training data into the classifier
opf.fit(X, Y)

# If data is labeled, one can assign predicted labels instead of only the cluster identifiers
opf.assign_labels()