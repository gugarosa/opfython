from opfython.models import KNNSupervisedOPF

# Creates a KNNSupervisedOPF instance
opf = KNNSupervisedOPF(max_k=10, distance='log_squared_euclidean', pre_computed_distance=None)
