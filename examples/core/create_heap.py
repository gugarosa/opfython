# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from opfython.core import Heap

h = Heap(size=5, policy="min")

h.insert(1)
n = h.remove()
