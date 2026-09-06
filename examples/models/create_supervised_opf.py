# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from opfython.models import SupervisedOPF

opf = SupervisedOPF(distance="log_squared_euclidean", pre_computed_distance=None)
