import numpy as np

import opfython.utils.constants as c
import opfython.utils.logging as l

logger = l.get_logger(__name__)


class Heap:
    """A standard implementation of a Heap structure.

    """

    def __init__(self, size=1):
        """Initialization method.

        Args:
            size (int): Maximum size of the heap.

        """

        logger.debug('Creating class: Heap.')

        # Maximum size of the heap
        self.size = size

        # List of costs
        self.cost = [0 for i in range(size)]

        # List of colors
        self.color = [c.WHITE for i in range(size)]

        # List of pixels
        self.p = [-1 for i in range(size)]

        # List of positioning markers
        self.pos = [-1 for i in range(size)]

        # Last element identifier
        self.last = -1

        logger.debug('Class created.')

    def is_full(self):
        """Checks if the heap is full.

        Returns:
            A boolean indicating whether the heap is full.

        """

        # If last position equals to size - 1
        if self.last == self.size - 1:
            # Return as True
            return True
        
        # If not, return as False
        return False

    def is_empty(self):
        """Checks if the heap is empty.

        Returns:
            A boolean indicating whether the heap is empty.

        """

        # If last position is equal to -1
        if self.last == -1:
            # Return as True
            return True

        # Return as False
        return False

    def dad(self, i):
        """Gathers the position of the dad's node.

        Returns:
            The position of dad's node.

        """

        # Returns the dad's position
        return int(((i - 1) / 2))

    def left_son(self, i):
        """Gathers the position of the left son's node.

        Returns:
            The position of left son's node.

        """

        # Returns the left son's position
        return int((2 * i + 1))

    def right_son(self, i):
        """Gathers the position of the right son's node.

        Returns:
            The position of right son's node.

        """

        # Return the right son's position
        return int((2 * i + 2))

    def go_up(self, i):
        """
        """

        # Gathers the dad's position
        j = self.dad(i)

        # While the heap exists and the cost of post-node is bigger than current node
        while ((i > 0) & (self.cost[self.p[j]] > self.cost[self.p[i]])):
            # Swap the positions
            self.p[j], self.p[i] = self.p[i], self.p[j]

            # Applies node's i value to the positioning list
            self.pos[self.p[i]] = i

            # Applies node's j value to the positioning list
            self.pos[self.p[j]] = j

            # Makes both indexes equal
            i = j

            # Gathers the new dad's position
            j = self.dad(i)

    def go_down(self, i):
        """
        """

        j = i
        left = self.left_son(i)
        right = self.right_son(i)

        if ((left <= self.last) & (self.cost[self.p[left]] < self.cost[self.p[i]])):
            j = left
        if ((right <= self.last) & (self.cost[self.p[right]] < self.cost[self.p[j]])):
            j = right

        if j is not i:
            self.p[j], self.p[i] = self.p[i], self.p[j]
            self.pos[self.p[i]] = i
            self.pos[self.p[j]] = j
            self.go_down(j)

    def insert(self, pixel):
        """
        """

        if not self.is_full():
            self.last += 1
            self.p[self.last] = pixel
            self.color[pixel] = c.GRAY
            self.pos[pixel] = self.last
            self.go_up(self.last)
            return True
        return False

    def remove(self):
        """
        """

        if not self.is_empty():
            pixel = self.p[0]
            self.pos[pixel] = -1
            self.color[pixel] = c.BLACK
            self.p[0] = self.p[self.last]
            self.pos[self.p[0]] = 0
            self.p[self.last] = -1
            self.last -= 1
            self.go_down(0)
            return pixel
        return False

    def update(self, p, color):
        """
        """

        self.cost[p] = color

        if self.color[p] == c.BLACK:
            pass

        if self.color[p] == c.WHITE:
            self.insert(p)
        else:
            self.go_up(self.pos[p])
