# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models import SupervisedOPF
from opfython.stream import loader

txt = loader.load_txt("data/boat.txt")
X, Y = p.parse_loader(txt)

X_train, X_val, Y_train, Y_val = s.split(X, Y, percentage=0.5, random_state=1)

opf = SupervisedOPF(distance="log_squared_euclidean", pre_computed_distance=None)

# Learning exchanges samples between these arrays in place
opf.learn(X_train, Y_train, X_val, Y_val, n_iterations=10)
