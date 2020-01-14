import numpy as np

import opfython.math.general as g
import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s
import opfython.utils.constants as c
from opfython.models.supervised import SupervisedOPF

def run_opf():
    # Fits training data into the classifier
    opf.fit(X_train, Y_train)

    # Predicts new data
    preds = opf.predict(X_val)

    # Calculating accuracy
    acc = g.opf_accuracy(Y_val, preds)

    return acc

# Loading a .txt file to a numpy array
txt = l.load_txt('data/boat.txt')

# Parsing a pre-loaded numpy array
X, Y = p.parse_array(txt)

# Splitting data into training and validation sets
X_train, X_val, Y_train, Y_val = s.split(
    X, Y, percentage=0.5, random_state=1)

# Defining input variables
max_iterations = 10
desired_acc = 0.9
i = 0

# Creates a SupervisedOPF instance
opf = SupervisedOPF(distance='log_squared_euclidean', pre_computed_distance=None)

# Running SupervisedOPF
current_acc = run_opf()

# Gathering current accuracy
acc = current_acc

while i < max_iterations and np.fabs(current_acc - acc) <= desired_acc:
    print(f'Running iteration {i+1}/{max_iterations} ...')

    # Gathering old accuracy as current accuracy
    acc = current_acc

    # Running SupervisedOPF
    run_opf()

    # Creating temporary lists
    X_temp, Y_temp = [], []

    # Removing irrelevant nodes
    for j, n in enumerate(opf.subgraph.nodes):
        if n.relevant != c.IRRELEVANT:
            X_temp.append(X_train[j, :])
            Y_temp.append(Y_train[j])

    # Copying lists back to original data
    X_train = np.asarray(X_temp)
    Y_train = np.asarray(Y_temp)
    
    # Running SupervisedOPF without irrelevant nodes
    current_acc = run_opf()

    print(f'Current accuracy: {current_acc} | Old accuracy: {acc}')

    # Incrementing iteration
    i += 1
