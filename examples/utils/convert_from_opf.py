# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import opfython.utils.converter as c

opf_file = "data/boat.dat"

c.opf2txt(opf_file, output_file="out.txt")
c.opf2csv(opf_file, output_file="out.csv")
c.opf2json(opf_file, output_file="out.json")
