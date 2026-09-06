# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import opfython.math.general as g
import opfython.stream.parser as p
from opfython.stream import loader

txt = loader.load_txt("data/boat.txt")
X, Y = p.parse_loader(txt)

g.pre_compute_distance(X, "boat_split_distances.txt", distance="log_squared_euclidean")
