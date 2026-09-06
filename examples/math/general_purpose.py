# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np

import opfython.math.general as g

array = np.asarray([1.5, 2, 0.5, 1.25, 1.75, 3])
labels = [0, 0, 0, 1, 1, 1, 2]
preds = [0, 0, 1, 1, 0, 1, 2]

norm_array = g.normalize(array)
print(norm_array)

c_matrix = g.confusion_matrix(labels, preds)
print(c_matrix)

opf_acc = g.opf_accuracy(labels, preds)
print(opf_acc)

opf_acc_per_label = g.opf_accuracy_per_label(labels, preds)
print(opf_acc_per_label)

purity = g.purity(labels, preds)
print(purity)
