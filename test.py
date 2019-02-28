from opfython.core.heap import Heap

size = 5

pathval = [0.0] * size

H = Heap(size=size)

print(H._cost)

pathval = [1.0] * size

H.insert(1)
H.remove()
print(H._last)
print(H._color)
print(H._pos)
print(H._pixel)