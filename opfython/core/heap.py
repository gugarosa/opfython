import numpy as np
import opfython.utils.logging as l

logger = l.get_logger(__name__)


class Heap:
    """

    Properties:

    Methods:

    """

    def __init__(self, size=1):
        """Initialization method.

        Args:

        """

        self.size = size
        self.cost = [0] * size
        self.color = ['WHITE'] * size
        self.pixel = [-1] * size
        self.pos = [-1] * size
        self.last = -1

    def is_full(self):
        if self.last == self.size - 1:
            return True
        return False

    def is_empty(self):
        if self.last == -1:
            return True
        return False

    def heap_dad(self, i):
        return int(((i - 1) / 2))

    def heap_left_son(self, i):
        return int((2 * i + 1))

    def heap_right_son(self, i):
        return int((2 * i + 2))

    def go_up(self, i):
        j = self.heap_dad(i)

        while ((i > 0) & (self.cost[self.pixel[j]] > self.cost[self.pixel[i]])):
            self.pixel[j], self.pixel[i] = self.pixel[i], self.pixel[j]
            self.pos[self.pixel[i]] = i
            self.pos[self.pixel[j]] = j
            i = j
            j = self.heap_dad(i)

    def go_down(self, i):
        j = i
        left = self.heap_left_son(i)
        right = self.heap_right_son(i)

        if ((left <= self.last) & (self.cost[self.pixel[left]] < self.cost[self.pixel[i]])):
            j = left
        if ((right <= self.last) & (self.cost[self.pixel[right]] < self.cost[self.pixel[j]])):
            j = right

        if j is not i:
            self.pixel[j], self.pixel[i] = self.pixel[i], self.pixel[j]
            self.pos[self.pixel[i]] = i
            self.pos[self.pixel[j]] = j
            self.go_down(j)

    def insert(self, pixel):
        if not self.is_full():
            self.last += 1
            self.pixel[self.last] = pixel
            self.color[pixel] = 'GRAY'
            self.pos[pixel] = self.last
            self.go_up(self.last)
            return True
        return False

    def remove(self):
        if not self.is_empty():
            pixel = self.pixel[0]
            self.pos[pixel] = -1
            self.color[pixel] = 'BLACK'
            self.pixel[0] = self.pixel[self.last]
            self.pos[self.pixel[0]] = 0
            self.pixel[self.last] = -1
            self.last -= 1
            self.go_down(0)
            return pixel
        return False

    def update(self, p, color):
        self.cost[p] = color

        if self.color[p] == 'BLACK':
            pass

        if self.color[p] == 'WHITE':
            self.insert(p)
        else:
            self.go_up(self.pos[p])