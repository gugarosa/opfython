from opfython.core import Heap

# Defining the maximum size of heap
size = 5

# Creating the heap
h = Heap(size=size, policy='min')

# Inserting a new node
h.insert(1)

# Removing the node
n = h.remove()
