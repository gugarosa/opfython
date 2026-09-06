# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import opfython.stream.parser as p
from opfython.stream import loader

txt = loader.load_txt("data/boat.txt")
X, Y = p.parse_loader(txt)
