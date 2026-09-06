# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import opfython.stream.parser as p
from opfython.stream import loader
from opfython.subgraphs import KNNSubgraph

input_file = "data/boat.txt"
txt = loader.load_txt(input_file)
X, Y = p.parse_loader(txt)

g = KNNSubgraph(X, Y)

# Direct file construction is useful when no intermediate transformation is needed
g = KNNSubgraph(from_file=input_file)
