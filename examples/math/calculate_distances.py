# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np

import opfython.math.distance as d

x = np.asarray([2, 3, 4, 5])
y = np.asarray([1, 2, 3, 1])

dist = d.euclidean_distance(x, y)

print(dist)
