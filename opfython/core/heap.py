# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Provide the mutable priority queue used by OPF algorithms."""

from typing import Literal

import opfython.utils.constants as c
import opfython.utils.exception as e


class Heap:
    """Maintain a fixed-size mutable min or max priority queue."""

    def __init__(self, size: int = 1, policy: str = "min") -> None:
        """Initialize an empty queue and its per-node bookkeeping.

        Args:
            size: Positive capacity and number of addressable node positions.
            policy: Minimum-first or maximum-first ordering selected by min or max.

        Raises:
            opfython.utils.exception.TypeError: The capacity is not a Python integer.
            opfython.utils.exception.ValueError: The capacity or ordering policy is invalid.

        """

        self.size = size
        self.policy = policy

        self.cost = [c.FLOAT_MAX] * size
        self.color = [c.WHITE] * size
        self.p = [-1] * size
        self.pos = [-1] * size

        self.last = -1

    @property
    def size(self) -> int:
        """Return the maximum number of queued nodes."""

        return self._size

    @size.setter
    def size(self, size: int) -> None:
        if not isinstance(size, int):
            raise e.TypeError(f"`size` should be an integer, but got {type(size).__name__}.")
        if size < 1:
            raise e.ValueError(f"`size` should be > 0, but got {size}.")

        self._size = size

    @property
    def policy(self) -> str:
        """Return the min or max ordering policy."""

        return self._policy

    @policy.setter
    def policy(self, policy: str) -> None:
        if policy not in ["min", "max"]:
            raise e.ValueError(f"`policy` should be min or max, but got {policy}.")

        self._policy = policy

    @property
    def cost(self) -> list[float]:
        """Return mutable priorities indexed by node position."""

        return self._cost

    @cost.setter
    def cost(self, cost: list[float]) -> None:
        if not isinstance(cost, list):
            raise e.TypeError(f"`cost` should be a list, but got {type(cost).__name__}.")

        self._cost = cost

    @property
    def color(self) -> list[int]:
        """Return each node's unseen, queued, or removed state."""

        return self._color

    @color.setter
    def color(self, color: list[int]) -> None:
        if not isinstance(color, list):
            raise e.TypeError(f"`color` should be a list, but got {type(color).__name__}.")

        self._color = color

    @property
    def p(self) -> list[int]:
        """Return the heap-position to node-position mapping."""

        return self._p

    @p.setter
    def p(self, p: list[int]) -> None:
        if not isinstance(p, list):
            raise e.TypeError(f"`p` should be a list, but got {type(p).__name__}.")

        self._p = p

    @property
    def pos(self) -> list[int]:
        """Return the node-position to heap-position mapping."""

        return self._pos

    @pos.setter
    def pos(self, pos: list[int]) -> None:
        if not isinstance(pos, list):
            raise e.TypeError(f"`pos` should be a list, but got {type(pos).__name__}.")

        self._pos = pos

    @property
    def last(self) -> int:
        """Return the last occupied heap position or -1 for an empty queue."""

        return self._last

    @last.setter
    def last(self, last: int) -> None:
        if not isinstance(last, int):
            raise e.TypeError(f"`last` should be an integer, but got {type(last).__name__}.")
        if last < -1:
            raise e.ValueError(f"`last` should be >= -1, but got {last}.")

        self._last = last

    def is_full(self) -> bool:
        """Report whether every heap slot is occupied.

        Returns:
            True when the queue has reached its capacity.

        """

        return self.last == self.size - 1

    def is_empty(self) -> bool:
        """Report whether the heap has no queued nodes.

        Returns:
            True when the last occupied position is -1.

        """

        return self.last == -1

    def dad(self, i: int) -> int:
        """Return a heap position's parent.

        Args:
            i: Position within the heap.

        Returns:
            Parent position, with the root mapped to itself.

        """

        return int((i - 1) / 2)

    def left_son(self, i: int) -> int:
        """Return a heap position's left child.

        Args:
            i: Position within the heap.

        Returns:
            Candidate left-child position, which can lie beyond the occupied queue.

        """

        return 2 * i + 1

    def right_son(self, i: int) -> int:
        """Return a heap position's right child.

        Args:
            i: Position within the heap.

        Returns:
            Candidate right-child position, which can lie beyond the occupied queue.

        """

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
        """Move a queued node toward the heap root.

        Args:
            i: Heap position whose priority may precede its parent.

        """

        parent = self.dad(i)
        while i > 0 and self._precedes(self.p[i], self.p[parent]):
            self._swap(i, parent)
            i = parent
            parent = self.dad(i)

    def go_down(self, i: int) -> None:
        """Move a queued node toward the heap leaves.

        Args:
            i: Heap position whose children may have earlier priorities.

        """

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
        """Insert a node if capacity is available.

        Args:
            p: Node position whose priority is already stored in cost.

        Returns:
            True when the node is inserted, or False when the queue is full.

        """

        if self.is_full():
            return False

        self.last += 1
        self.p[self.last] = p
        self.color[p] = c.GRAY
        self.pos[p] = self.last
        self.go_up(self.last)

        return True

    def remove(self) -> int | Literal[False]:
        """Remove the next node according to the ordering policy.

        Returns:
            Removed node position, or False when the queue is empty.

        """

        if self.is_empty():
            return False

        node = self.p[0]
        self.color[node] = c.BLACK

        self.p[0] = self.p[self.last]
        self.pos[self.p[0]] = 0
        self.pos[node] = -1
        self.p[self.last] = -1
        self.last -= 1

        self.go_down(0)

        return node

    def update(self, p: int, cost: float) -> None:
        """Update a node's priority and move it toward the root.

        OPF callers use this operation to improve a priority under the current
        policy. An unseen node is inserted into the queue.

        Args:
            p: Node position whose priority is changing.
            cost: Updated priority value.

        """

        self.cost[p] = cost

        if self.color[p] == c.WHITE:
            self.insert(p)
        else:
            self.go_up(self.pos[p])
