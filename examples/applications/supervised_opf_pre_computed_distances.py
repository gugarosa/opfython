# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import opfython.math.general as g
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models import SupervisedOPF
from opfython.stream import loader

txt = loader.load_txt("data/boat.txt")
X, Y = p.parse_loader(txt)

g.pre_compute_distance(X, "data/boat_distances.txt", distance="log_squared_euclidean")

# Preserve original matrix indexes when rearranging the samples
X_train, X_test, Y_train, Y_test, I_train, I_test = s.split_with_index(X, Y, percentage=0.5, random_state=1)

opf = SupervisedOPF(distance="log_squared_euclidean", pre_computed_distance="data/boat_distances.txt")

opf.fit(X_train, Y_train, I_train)

preds = opf.predict(X_test, I_test)
acc = g.opf_accuracy(Y_test, preds)

print(f"Accuracy: {acc}")
