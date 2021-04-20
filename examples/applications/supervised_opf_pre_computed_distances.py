import opfython.math.general as g
import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models import SupervisedOPF

# Loading a .txt file to a numpy array
txt = l.load_txt('data/boat.txt')

# Parsing a pre-loaded numpy array
X, Y = p.parse_loader(txt)

# Pre-computing the distances
g.pre_compute_distance(X, 'data/boat_distances.txt', distance='log_squared_euclidean')

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test, I_train, I_test = s.split_with_index(X, Y, percentage=0.5, random_state=1)

# Creates a SupervisedOPF instance
opf = SupervisedOPF(distance='log_squared_euclidean', pre_computed_distance='data/boat_distances.txt')

# Fits training data into the classifier
opf.fit(X_train, Y_train, I_train)

# Predicts new data
preds = opf.predict(X_test, I_test)

# Calculating accuracy
acc = g.opf_accuracy(Y_test, preds)

print(f'Accuracy: {acc}')
