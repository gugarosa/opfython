"""Priority queue used by OPF algorithms."""

from typing import List

import opfython.utils.constants as c
import opfython.utils.exception as e


class Heap:
    """Fixed-size mutable min or max heap."""

    def __init__(self, size: int = 1, policy: str = "min") -> None:
        self.size = size
        self.policy = policy
        self.cost = [c.FLOAT_MAX] * size
        self.color = [c.WHITE] * size
        self.p = [-1] * size
        self.pos = [-1] * size
        self.last = -1

    @property
    def size(self) -> int:
        """Maximum heap size."""

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
        """Heap ordering policy."""

        return self._policy

    @policy.setter
    def policy(self, policy: str) -> None:
        if policy not in ["min", "max"]:
            raise e.ValueError("`policy` should be `min` or `max`")
        self._policy = policy

    @property
    def cost(self) -> List[float]:
        """Node costs."""

        return self._cost

    @cost.setter
    def cost(self, cost: List[float]) -> None:
        if not isinstance(cost, list):
            raise e.TypeError("`cost` should be a list")
        self._cost = cost

    @property
    def color(self) -> List[int]:
        """Node colors."""

        return self._color

    @color.setter
    def color(self, color: List[int]) -> None:
        if not isinstance(color, list):
            raise e.TypeError("`color` should be a list")
        self._color = color

    @property
    def p(self) -> List[int]:
        """Heap-position to node mapping."""

        return self._p

    @p.setter
    def p(self, p: List[int]) -> None:
        if not isinstance(p, list):
            raise e.TypeError("`p` should be a list")
        self._p = p

    @property
    def pos(self) -> List[int]:
        """Node to heap-position mapping."""

        return self._pos

    @pos.setter
    def pos(self, pos: List[int]) -> None:
        if not isinstance(pos, list):
            raise e.TypeError("`pos` should be a list")
        self._pos = pos

    @property
    def last(self) -> int:
        """Last occupied heap position."""

        return self._last

    @last.setter
    def last(self, last: int) -> None:
        if not isinstance(last, int):
            raise e.TypeError("`last` should be an integer")
        if last < -1:
            raise e.ValueError("`last` should be > -1")
        self._last = last

    def is_full(self) -> bool:
        """Return whether every heap slot is occupied."""

        return self.last == self.size - 1

    def is_empty(self) -> bool:
        """Return whether the heap has no nodes."""

        return self.last == -1

    def dad(self, i: int) -> int:
        """Return a node's parent position."""

        return int((i - 1) / 2)

    def left_son(self, i: int) -> int:
        """Return a node's left-child position."""

        return 2 * i + 1

    def right_son(self, i: int) -> int:
        """Return a node's right-child position."""

        return 2 * i + 2

    def _precedes(self, left: int, right: int) -> bool:
        if self.policy == "min":
            return self.cost[left] < self.cost[right]
        return self.cost[left] > self.cost[right]

    def _swap(self, left: int, right: int) -> None:
        self.p[left], self.p[right] = self.p[right], self.p[left]
        self.pos[self.p[left]] = left
        self.pos[self.p[right]] = right

    def go_up(self, i: int) -> None:
        """Move a node toward the heap root."""

        parent = self.dad(i)
        while i > 0 and self._precedes(self.p[i], self.p[parent]):
            self._swap(i, parent)
            i = parent
            parent = self.dad(i)

    def go_down(self, i: int) -> None:
        """Move a node toward the heap leaves."""

        left = self.left_son(i)
        right = self.right_son(i)
        target = i

        if left <= self.last and self._precedes(self.p[left], self.p[target]):
            target = left
        if right <= self.last and self._precedes(self.p[right], self.p[target]):
            target = right

        if target != i:
            self._swap(i, target)
            self.go_down(target)

    def insert(self, p: int) -> bool:
        """Insert a node if capacity is available."""

        if self.is_full():
            return False

        self.last += 1
        self.p[self.last] = p
        self.color[p] = c.GRAY
        self.pos[p] = self.last
        self.go_up(self.last)
        return True

    def remove(self) -> int:
        """Remove and return the next node, or ``False`` when empty."""

        if self.is_empty():
            return False

        node = self.p[0]
        self.pos[node] = -1
        self.color[node] = c.BLACK
        self.p[0] = self.p[self.last]
        self.pos[self.p[0]] = 0
        self.p[self.last] = -1
        self.last -= 1
        self.go_down(0)
        return node

    def update(self, p: int, cost: float) -> None:
        """Update a node's cost, inserting it when unseen."""

        self.cost[p] = cost
        if self.color[p] == c.BLACK:
            pass

        if self.color[p] == c.WHITE:
            self.insert(p)
        else:
            self.go_up(self.pos[p])
