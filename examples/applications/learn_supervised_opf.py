import copy

import numpy as np

import opfython.math.general as g
import opfython.math.random as r
import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s
import opfython.utils.constants as c
from opfython.core.subgraph import Subgraph
from opfython.models.supervised import SupervisedOPF

# Loading a .txt file to a numpy array
txt = l.load_txt('data/boat.txt')

# Parsing a pre-loaded numpy array
X, Y = p.parse_array(txt)

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = s.split(
    X, Y, percentage=0.5, random_state=1)

# Defining initial variables
max_acc = 0
previous_acc = 0
i = 0
n_iterations = 10

# Creates a always true loop
while True:
    print(f'Running iteration {i+1}/{n_iterations} ...')

    # Creates a SupervisedOPF instance
    opf = SupervisedOPF(distance='log_squared_euclidean',
                        pre_computed_distance=None)

    # Fits training data into the classifier
    opf.fit(X_train, Y_train)

    # Predicts new data
    preds = opf.predict(X_test)

    # Calculating accuracy
    acc = g.opf_accuracy(Y_test, preds)

    # Checks if current accuracy is better than the best one
    if acc > max_acc:
        # If yes, replace the maximum accuracy
        max_acc = acc

        # And makes a copy of the best OPF classifier
        best_opf = copy.deepcopy(opf)

    # Gathers which samples were missclassified
    errors = np.argwhere(Y_test != preds)

    # Defining the initial number of non-prototypes as 0
    non_prototypes = 0

    # For every possible subgraph's node
    for n in opf.subgraph.nodes:
        # If the node is not a prototype
        if n.status != c.PROTOTYPE:
            # Increments the number of non-prototypes
            non_prototypes += 1

    # For every possible error
    for e in errors:
        # Counter will receive the number of non-prototypes
        ctr = non_prototypes

        # While the counter is bigger than zero
        while ctr > 0:
            # Generates a random index
            j = int(r.generate_uniform_random_number(0, len(X_train)))

            # If the node on that particular index is not a prototype
            if opf.subgraph.nodes[j].status != c.PROTOTYPE:
                # Swap the nodes
                X_train[j, :], X_test[e, :] = X_test[e, :], X_train[j, :]
                Y_train[j], Y_test[e] = Y_test[e], Y_train[j]

                # Decrements the number of non-prototypes
                non_prototypes -= 1

                # Resets the counter
                ctr = 0

            # If the node on that particular index is a prototype
            else:
                # Decrements the counter
                ctr -= 1

    # Calculating difference between current accuracy and previous one
    delta = np.fabs(acc - previous_acc)

    # Replacing the previous accuracy as current accuracy
    previous_acc = acc

    # Incrementing the counter
    i += 1

    print(f'Accuracy: {acc} | Delta: {delta} | Maximum Accuracy: {max_acc}')

    # If the difference is smaller than 10e-4 or iterations are finished
    if delta < 0.0001 and i == n_iterations:
        # Breaks the loop
        break
