import numpy as np

import opfython.math.general as g
import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.core.subgraph import Subgraph
from opfython.models.supervised import SupervisedOPF

# Loading a .txt file to a numpy array
txt = l.load_txt('data/boat.txt')

# Parsing a pre-loaded numpy array
X, Y = p.parse_array(txt)

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = s.split(
    X, Y, percentage=0.5, random_state=2)

# Creates a always true loop
while True:
    # Creates a SupervisedOPF instance
    opf = SupervisedOPF(distance='log_squared_euclidean',
                        pre_computed_distance=None)

    # Fits training data into the classifier
    opf.fit(X_train, Y_train)

    # Predicts new data
    preds = opf.predict(X_test)

    # Calculating accuracy
    acc = g.opf_accuracy(Y_test, preds)

    print(f'Accuracy: {acc}')

    # Gathers which samples were missclassified
    errors = np.argwhere(Y_test != preds)

    # If there are no missclassified samples
    if len(errors) == 0:
        # Breaks the process
        break

    # For every wrong classified sample
    for e in errors:
        # Adds the sample to the training set
        X_train = np.vstack((X_train, X_test[e, :]))
        Y_train = np.hstack((Y_train, Y_test[e]))

    # For every wrong classified sample
    for e in errors:
        # Deletes the sample from the testing set
        X_test = np.delete(X_test, e, axis=0)
        Y_test = np.delete(Y_test, e)
