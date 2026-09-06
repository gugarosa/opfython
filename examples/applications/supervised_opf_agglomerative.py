# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np

import opfython.math.general as g
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models import SupervisedOPF
from opfython.stream import loader

txt = loader.load_txt("data/boat.txt")
X, Y = p.parse_loader(txt)

X_train, X_val, Y_train, Y_val = s.split(X, Y, percentage=0.5, random_state=1)

# Adding validation mistakes can change which training nodes become prototypes
while True:
    opf = SupervisedOPF(distance="log_squared_euclidean", pre_computed_distance=None)
    opf.fit(X_train, Y_train)

    preds = opf.predict(X_val)
    acc = g.opf_accuracy(Y_val, preds)

    print(f"Accuracy: {acc}")

    errors = np.argwhere(Y_val != preds)
    if len(errors) == 0:
        break

    for e in errors:
        X_train = np.vstack((X_train, X_val[e, :]))
        Y_train = np.hstack((Y_train, Y_val[e]))

    for e in errors:
        X_val = np.delete(X_val, e, axis=0)
        Y_val = np.delete(Y_val, e)
