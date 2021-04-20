import opfython.math.general as g
import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models import KNNSupervisedOPF

# Loading a .txt file to a numpy array
txt = l.load_txt('data/boat.txt')

# Parsing a pre-loaded numpy array
X, Y = p.parse_loader(txt)

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = s.split(
    X, Y, percentage=0.8, random_state=1)

# Splitting data into training and validation sets
X_train, X_val, Y_train, Y_val = s.split(
    X_train, Y_train, percentage=0.25, random_state=1)

# Creates an KNNSupervisedOPF instance
opf = KNNSupervisedOPF(max_k=10, distance='log_squared_euclidean', pre_computed_distance=None)

# Fits training data into the classifier
opf.fit(X_train, Y_train, X_val, Y_val)

# Predicts new data
preds = opf.predict(X_test)

# Calculating accuracy
acc = g.opf_accuracy(Y_test, preds)

print(f'Accuracy: {acc}')
