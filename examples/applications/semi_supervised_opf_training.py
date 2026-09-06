# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import opfython.math.general as g
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models import SemiSupervisedOPF
from opfython.stream import loader

txt = loader.load_txt("data/boat.txt")
X, Y = p.parse_loader(txt)

X_train, X_test, Y_train, Y_test = s.split(X, Y, percentage=0.8, random_state=1)

# Withhold these labels to demonstrate semi-supervised training
X_train, X_unlabeled, Y_train, Y_unlabeled = s.split(X_train, Y_train, percentage=0.25, random_state=1)

opf = SemiSupervisedOPF(distance="log_squared_euclidean", pre_computed_distance=None)
opf.fit(X_train, Y_train, X_unlabeled)

preds = opf.predict(X_test)
acc = g.opf_accuracy(Y_test, preds)

print(f"Accuracy: {acc}")
