"""Standard Heap implementation.
"""

from typing import List, Optional

import opfython.utils.constants as c
import opfython.utils.exception as e


class Heap:
    """A standard implementation of a Heap structure."""

    def __init__(self, size: int = 1, policy: str = "min") -> None:
        """Initialization method.

        Args:
            size: Maximum size of the heap.
            policy: Heap's policy (`min` or `max`).

        """

        self.size = size
        self.policy = policy

        self.cost = [c.FLOAT_MAX for i in range(size)]
        self.color = [c.WHITE for i in range(size)]
        self.p = [-1 for i in range(size)]
        self.pos = [-1 for i in range(size)]

        self.last = -1

    @property
    def size(self) -> int:
        """Maximum size of the heap."""

        return self._size

    @size.setter
    def size(self, size: int) -> None:
        if not isinstance(size, int):
            raise e.TypeError("`size` should be an integer")
        if size < 1:
            raise e.ValueError("`size` should be > 0")

        self._size = size

    @property
    def policy(self) -> str:
        """Policy that rules the heap."""

        return self._policy

    @policy.setter
    def policy(self, policy: str) -> None:
        if policy not in ["min", "max"]:
            raise e.ValueError("`policy` should be `min` or `max`")

        self._policy = policy

    @property
    def cost(self) -> List[float]:
        """List of nodes' costs."""

        return self._cost

    @cost.setter
    def cost(self, cost: List[float]) -> None:
        if not isinstance(cost, list):
            raise e.TypeError("`cost` should be a list")

        self._cost = cost

    @property
    def color(self) -> List[int]:
        """List of nodes' colors."""

        return self._color

    @color.setter
    def color(self, color: List[int]) -> None:
        if not isinstance(color, list):
            raise e.TypeError("`color` should be a list")

        self._color = color

    @property
    def p(self) -> List[int]:
        """List of nodes' values."""

        return self._p

    @p.setter
    def p(self, p: List[int]) -> None:
        if not isinstance(p, list):
            raise e.TypeError("`p` should be a list")

        self._p = p

    @property
    def pos(self) -> List[int]:
        """List of nodes' positioning markers."""

        return self._pos

    @pos.setter
    def pos(self, pos: List[int]) -> None:
        if not isinstance(pos, list):
            raise e.TypeError("`pos` should be a list")

        self._pos = pos

    @property
    def last(self) -> int:
        """Last element identifier."""

        return self._last

    @last.setter
    def last(self, last: int) -> None:
        if not isinstance(last, int):
            raise e.TypeError("`last` should be an integer")
        if last < -1:
            raise e.ValueError("`last` should be > -1")

        self._last = last

    def is_full(self) -> bool:
        """Checks if the heap is full.

        Returns:
            (bool): A boolean indicating whether the heap is full.

        """

        if self.last == (self.size - 1):
            return True

        return False

    def is_empty(self) -> bool:
        """Checks if the heap is empty.

        Returns:
            (bool): A boolean indicating whether the heap is empty.

        """

        if self.last == -1:
            return True

        return False

    def dad(self, i: int) -> int:
        """Gathers the position of the node's dad.

        Args:
            i: Node's position.

        Returns:
            (int): The position of node's dad.

        """

        return int(((i - 1) / 2))

    def left_son(self, i: int) -> int:
        """Gathers the position of the node's left son.

        Args:
            i: Node's position.

        Returns:
            (int): The position of node's left son

        """

        return int((2 * i + 1))

    def right_son(self, i: int) -> int:
        """Gathers the position of the node's right son.

        Args:
            i: Node's position.

        Returns:
            (int): The position of node's right son.

        """

        return int((2 * i + 2))

    def go_up(self, i: int) -> None:
        """Goes up in the heap.

        Args:
            i: Position to be achieved.

        """

        j = self.dad(i)

        if self.policy == "min":
            # While the heap exists and the cost of post-node is bigger than current node
            while i > 0 and self.cost[self.p[j]] > self.cost[self.p[i]]:
                self.p[j], self.p[i] = self.p[i], self.p[j]

                self.pos[self.p[i]] = i
                self.pos[self.p[j]] = j

                i = j
                j = self.dad(i)

        else:
            # While the heap exists and the cost of post-node is smaller than current node
            while i > 0 and self.cost[self.p[j]] < self.cost[self.p[i]]:
                self.p[j], self.p[i] = self.p[i], self.p[j]

                self.pos[self.p[i]] = i
                self.pos[self.p[j]] = j

                i = j
                j = self.dad(i)

    def go_down(self, i: int) -> None:
        """Goes down in the heap.

        Args:
            i: Position to be achieved.

        """

        left = self.left_son(i)
        right = self.right_son(i)

        j = i

        if self.policy == "min":
            # Checks if left node is not the last and its cost is smaller than previous
            if left <= self.last and self.cost[self.p[left]] < self.cost[self.p[i]]:
                j = left

            # Checks if right node is not the last and its cost is smaller than previous
            if right <= self.last and self.cost[self.p[right]] < self.cost[self.p[j]]:
                j = right

        else:
            # Checks if left node is not the last and its cost is bigger than previous
            if left <= self.last and self.cost[self.p[left]] > self.cost[self.p[i]]:
                j = left

            # Checks if right node is not the last and its cost is bigger than previous
            if right <= self.last and self.cost[self.p[right]] > self.cost[self.p[j]]:
                j = right

        if j != i:
            self.p[j], self.p[i] = self.p[i], self.p[j]

            self.pos[self.p[i]] = i
            self.pos[self.p[j]] = j

            self.go_down(j)

    def insert(self, p: int) -> bool:
        """Inserts a new node into the heap.

        Args:
            p: Node's value to be inserted.

        Returns:
            (bool): Boolean indicating whether insertion was performed correctly.

        """

        if not self.is_full():
            self.last += 1

            self.p[self.last] = p
            self.color[p] = c.GRAY
            self.pos[p] = self.last

            self.go_up(self.last)

            return True

        return False

    def remove(self) -> int:
        """Removes a node from the heap.

        Returns:
            (int): The removed node value.

        """

        if not self.is_empty():
            p = self.p[0]

            self.pos[p] = -1
            self.color[p] = c.BLACK

            self.p[0] = self.p[self.last]

            self.pos[self.p[0]] = 0
            self.p[self.last] = -1

            self.last -= 1

            self.go_down(0)

            return p

        return False

    def update(self, p: int, cost: float) -> None:
        """Updates a node with a new value.

        Args:
            p: Node's position.
            cost: Node's cost.

        """

        self.cost[p] = cost

        if self.color[p] == c.BLACK:
            pass

        if self.color[p] == c.WHITE:
            self.insert(p)
        else:
            self.go_up(self.pos[p])
