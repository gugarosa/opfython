# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.stream import loader

txt = loader.load_txt("data/boat.txt")
X, Y = p.parse_loader(txt)

X_train, X_test, Y_train, Y_test = s.split(X, Y, percentage=0.5, random_state=1)

X, Y = s.merge(X_train, X_test, Y_train, Y_test)
