import opfython.math.general as g
import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models import UnsupervisedOPF

# Loading a .txt file to a numpy array
txt = l.load_txt('data/boat.txt')

# Parsing a pre-loaded numpy array
X, Y = p.parse_loader(txt)

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = s.split(
    X, Y, percentage=0.5, random_state=1)

# Creates an UnsupervisedOPF instance
opf = UnsupervisedOPF(
    min_k=1, max_k=10, distance='log_squared_euclidean', pre_computed_distance=None)

# Fits training data into the classifier
opf.fit(X_train, Y_train)

# If data is labeled, one can propagate predicted labels instead of only the cluster identifiers
opf.propagate_labels()

# Predicts new data
preds, clusters = opf.predict(X_test)

# Calculating accuracy
acc = g.opf_accuracy(Y_test, preds)

print(f'Accuracy: {acc}')
