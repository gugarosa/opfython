import opfython.math.general as g
import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models.knn_supervised import KNNSupervisedOPF

# Loading a .txt file to a numpy array
txt = l.load_txt('data/boat.txt')

# Parsing a pre-loaded numpy array
X, Y = p.parse_array(txt)

# # Splitting data into training and validating sets
# X_train, X_val, Y_train, Y_val = s.split(
#     X, Y, percentage=0.5, random_state=1)

# Creates an KNNSupervisedOPF instance
opf = KNNSupervisedOPF(max_k=10, distance='log_squared_euclidean', pre_computed_distance=None)

# Fits training data into the classifier
opf.fit(X, Y, X, Y)
