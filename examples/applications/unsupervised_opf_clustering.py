# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import opfython.math.general as g
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models import UnsupervisedOPF
from opfython.stream import loader

txt = loader.load_txt("data/boat.txt")
X, Y = p.parse_loader(txt)

X_train, X_test, Y_train, Y_test = s.split(X, Y, percentage=0.5, random_state=1)

opf = UnsupervisedOPF(min_k=1, max_k=10, distance="log_squared_euclidean", pre_computed_distance=None)

opf.fit(X_train, Y_train)

# Root labels are propagated separately from the unsupervised cluster assignments
opf.propagate_labels()

preds, clusters = opf.predict(X_test)
acc = g.opf_accuracy(Y_test, preds)

print(f"Accuracy: {acc}")
