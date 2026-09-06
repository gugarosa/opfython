# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import opfython.stream.parser as p
from opfython.core import Subgraph
from opfython.stream import loader

input_file = "data/boat.txt"
txt = loader.load_txt(input_file)
X, Y = p.parse_loader(txt)

g = Subgraph(X, Y)

# Direct file construction is useful when no intermediate transformation is needed
g = Subgraph(from_file=input_file)
