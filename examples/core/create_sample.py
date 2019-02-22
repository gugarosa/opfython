import numpy as np

from opfython.core.sample import Sample

# Creating features vector
x = np.zeros(10)

# Instanciating a Sample
s = Sample(label=1, features=x)