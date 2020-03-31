import sys

# A constant value used to avoid division by zero, zero logarithms
# and any possible mathematical error
EPSILON = 1e-10

# When the costs are initialized, their value are defined as
# the maximum float value possible
FLOAT_MAX = sys.float_info.max

# Defining color constants for the Heap structure
# Note that these constants should not be modified
WHITE = 0
GRAY = 1
BLACK = 2

# Defining constant to identify whether a node in
# the subgraph has a predecessor or not
NIL = -1

# Defining constant to identify whether a node is
# a prototype or not
STANDARD = 0
PROTOTYPE = 1

# Defining constant to identify whether a node is
# relevant or not
IRRELEVANT = 0
RELEVANT = 1

# Defining constant to reflect the maximum arc weight
# used to calculate the distance measures
MAX_ARC_WEIGHT = 100000

# Defining constant to reflect the maximum density
# used to calculate in unsupervised approaches
MAX_DENSITY = 1000
