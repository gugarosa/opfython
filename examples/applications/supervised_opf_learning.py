import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models import SupervisedOPF

# Loading a .txt file to a numpy array
txt = l.load_txt('data/boat.txt')

# Parsing a pre-loaded numpy array
X, Y = p.parse_loader(txt)

# Splitting data into training and validation sets
X_train, X_val, Y_train, Y_val = s.split(
    X, Y, percentage=0.5, random_state=1)

# Creates a SupervisedOPF instance
opf = SupervisedOPF(distance='log_squared_euclidean',
                    pre_computed_distance=None)

# Performs the learning procedure
opf.learn(X_train, Y_train, X_val, Y_val, n_iterations=10)
