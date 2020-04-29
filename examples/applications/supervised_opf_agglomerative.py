import numpy as np

import opfython.math.general as g
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

# Creates a always true loop
while True:
    # Creates a SupervisedOPF instance
    opf = SupervisedOPF(distance='log_squared_euclidean',
                        pre_computed_distance=None)

    # Fits training data into the classifier
    opf.fit(X_train, Y_train)

    # Predicts new data
    preds = opf.predict(X_val)

    # Calculating accuracy
    acc = g.opf_accuracy(Y_val, preds)

    print(f'Accuracy: {acc}')

    # Gathers which samples were missclassified
    errors = np.argwhere(Y_val != preds)

    # If there are no missclassified samples
    if len(errors) == 0:
        # Breaks the process
        break

    # For every wrong classified sample
    for e in errors:
        # Adds the sample to the training set
        X_train = np.vstack((X_train, X_val[e, :]))
        Y_train = np.hstack((Y_train, Y_val[e]))

    # For every wrong classified sample
    for e in errors:
        # Deletes the sample from the testing set
        X_val = np.delete(X_val, e, axis=0)
        Y_val = np.delete(Y_val, e)
