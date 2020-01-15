import pytest
from opfython.core import heap


def test_heap_size():
    h = heap.Heap()

    assert h.size == 1


def test_heap_size_setter():
    h = heap.Heap()

    try:
        h.size = 1.15
    except:
        h.size = 1

    assert h.size == 1

    try:
        h.size = 0
    except:
        h.size = 1

    assert h.size == 1


def test_heap_policy():
    h = heap.Heap()

    assert h.policy == 'min'


def test_heap_policy_setter():
    h = heap.Heap()

    try:
        h.policy = 'a'
    except:
        h.policy = 'min'

    assert h.policy == 'min'

    try:
        h.policy = 'b'
    except:
        h.policy = 'max'

    assert h.policy == 'max'
