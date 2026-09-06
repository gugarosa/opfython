# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from opfython.models import UnsupervisedOPF

opf = UnsupervisedOPF(min_k=1, max_k=10, distance="log_squared_euclidean", pre_computed_distance=None)
