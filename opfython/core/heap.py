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

        self._size = size
        self._cost = [0] * size
        self._color = ['WHITE'] * size
        self._pixel = [-1] * size
        self._pos = [-1] * size
        self._last = -1

    def is_full(self):
        if self._last == self._size - 1:
            return True
        return False

    def is_empty(self):
        if self._last == -1:
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

        while ((i > 0) & (self._cost[self._pixel[j]] > self._cost[self._pixel[i]])):
            self._pixel[j], self._pixel[i] = self._pixel[i], self._pixel[j]
            self._pos[self._pixel[i]] = i
            self._pos[self._pixel[j]] = j
            i = j
            j = self.heap_dad(i)

    def go_down(self, i):
        j = i
        left = self.heap_left_son(i)
        right = self.heap_right_son(i)

        if ((left <= self._last) & (self._cost[self._pixel[left]] < self._cost[self._pixel[i]])):
            j = left
        if ((right <= self._last) & (self._cost[self._pixel[right]] < self._cost[self._pixel[j]])):
            j = right

        if j is not i:
            self._pixel[j], self._pixel[i] = self._pixel[i], self._pixel[j]
            self._pos[self._pixel[i]] = i
            self._pos[self._pixel[j]] = j
            self.go_down(j)

    def insert(self, pixel):
        if not self.is_full():
            self._last += 1
            self._pixel[self._last] = pixel
            self._color[pixel] = 'GRAY'
            self._pos[pixel] = self._last
            self.go_up(self._last)
            return True
        return False

    def remove(self):
        if not self.is_empty():
            pixel = self._pixel[0]
            self._pos[pixel] = -1
            self._color[pixel] = 'BLACK'
            self._pixel[0] = self._pixel[self._last]
            self._pos[self._pixel[0]] = 0
            self._pixel[self._last] = -1
            self._last -= 1
            self.go_down(0)
            return pixel
        return False
