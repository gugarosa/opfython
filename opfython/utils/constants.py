# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Constants."""

import sys

# Regularize denominators and logarithm arguments without changing distance formulas
EPSILON = 1e-20

FLOAT_MAX = sys.float_info.max

WHITE = 0
GRAY = 1
BLACK = 2

NIL = -1

STANDARD = 0
PROTOTYPE = 1

IRRELEVANT = 0
RELEVANT = 1

MAX_ARC_WEIGHT = 100000

MAX_DENSITY = 1000
