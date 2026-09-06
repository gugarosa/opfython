# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from opfython.stream import loader

csv = loader.load_csv("data/boat.csv")
txt = loader.load_txt("data/boat.txt")
json = loader.load_json("data/boat.json")
