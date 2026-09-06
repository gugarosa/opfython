# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import opfython.math.random as r

u = r.generate_uniform_random_number(low=0.0, high=1.0, size=10)
print(u)

g = r.generate_gaussian_random_number(mean=0.5, variance=1.0, size=10)
print(g)
