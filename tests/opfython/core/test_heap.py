import pytest

from opfython.core import Heap
from opfython.utils import exception


def test_heap_min_policy_orders_updates():
    heap = Heap(size=3)

    heap.update(0, 3)
    heap.update(1, 1)
    heap.update(2, 2)
    heap.update(0, 0)

    assert heap.is_full()
    assert [heap.remove(), heap.remove(), heap.remove()] == [0, 1, 2]
    assert heap.is_empty()
    assert heap.remove() is False


def test_heap_max_policy_orders_updates():
    heap = Heap(size=3, policy="max")

    heap.update(0, 1)
    heap.update(1, 3)
    heap.update(2, 2)

    assert [heap.remove(), heap.remove(), heap.remove()] == [1, 2, 0]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"size": 1.5}, exception.TypeError),
        ({"size": 0}, exception.ValueError),
        ({"policy": "unknown"}, exception.ValueError),
        ({"policy": []}, exception.ValueError),
    ],
)
def test_heap_rejects_invalid_configuration(kwargs, error):
    with pytest.raises(error):
        Heap(**kwargs)


@pytest.mark.parametrize(
    ("attribute", "value", "error"),
    [
        ("size", 1.5, exception.TypeError),
        ("size", 0, exception.ValueError),
        ("policy", "unknown", exception.ValueError),
        ("policy", [], exception.ValueError),
        ("cost", "invalid", exception.TypeError),
        ("color", "invalid", exception.TypeError),
        ("p", "invalid", exception.TypeError),
        ("pos", "invalid", exception.TypeError),
        ("last", 1.5, exception.TypeError),
        ("last", -2, exception.ValueError),
    ],
)
def test_heap_validates_public_attributes(attribute, value, error):
    heap = Heap()

    with pytest.raises(error):
        setattr(heap, attribute, value)


def test_heap_navigation_helpers():
    heap = Heap(size=10)

    assert heap.dad(5) == 2
    assert heap.left_son(5) == 11
    assert heap.right_son(5) == 12


@pytest.mark.parametrize("policy", ["min", "max"])
@pytest.mark.parametrize("size", [1, 3])
def test_heap_removal_clears_position(policy, size):
    heap = Heap(size=size, policy=policy)
    for node in range(size):
        heap.update(node, node)

    while not heap.is_empty():
        removed = heap.remove()
        assert heap.pos[removed] == -1
        for position in range(heap.last + 1):
            assert heap.pos[heap.p[position]] == position
