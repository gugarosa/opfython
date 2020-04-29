from opfython.models import UnsupervisedOPF

# Creates an UnsupervisedOPF instance
opf = UnsupervisedOPF(min_k=1, max_k=10, distance='log_squared_euclidean', pre_computed_distance=None)
