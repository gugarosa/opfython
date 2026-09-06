# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np

from opfython.core import Node

idx = 0
label = 0
features = np.asarray([2, 2.5, 1.5, 4])

n = Node(idx, label, features)
