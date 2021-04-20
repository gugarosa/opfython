import numpy as np

import opfython.math.general as g

# Defining array, labels and predictions
array = np.asarray([1.5, 2, 0.5, 1.25, 1.75, 3])
labels = [0, 0, 0, 1, 1, 1, 2]
preds = [0, 0, 1, 1, 0, 1, 2]

# Normalizing the array
norm_array = g.normalize(array)
print(norm_array)

# Calculating the confusion matrix
c_matrix = g.confusion_matrix(labels, preds)
print(c_matrix)

# Calculating OPF-like accuracy
opf_acc = g.opf_accuracy(labels, preds)
print(opf_acc)

# Calculating OPF-like accuracy per label
opf_acc_per_label = g.opf_accuracy_per_label(labels, preds)
print(opf_acc_per_label)

# Calculating purity measure
purity = g.purity(labels, preds)
print(purity)
