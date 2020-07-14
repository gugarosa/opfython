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


def test_heap_cost():
    h = heap.Heap()

    assert len(h.cost) == 1


def test_heap_cost_setter():
    h = heap.Heap()

    try:
        h.cost = 'a'
    except:
        h.cost = []

    assert isinstance(h.cost, list)


def test_heap_color():
    h = heap.Heap()

    assert len(h.color) == 1


def test_heap_color_setter():
    h = heap.Heap()

    try:
        h.color = 'a'
    except:
        h.color = []

    assert isinstance(h.color, list)


def test_heap_p():
    h = heap.Heap()

    assert len(h.p) == 1


def test_heap_p_setter():
    h = heap.Heap()

    try:
        h.p = 'a'
    except:
        h.p = []

    assert isinstance(h.p, list)


def test_heap_pos():
    h = heap.Heap()

    assert len(h.pos) == 1


def test_heap_pos_setter():
    h = heap.Heap()

    try:
        h.pos = 'a'
    except:
        h.pos = []

    assert isinstance(h.pos, list)


def test_heap_last():
    h = heap.Heap()

    assert h.last == -1


def test_heap_last_setter():
    h = heap.Heap()

    try:
        h.last = 10.5
    except:
        h.last = -1

    assert h.last == -1

    try:
        h.last = -2
    except:
        h.last = -1

    assert h.last == -1


def test_heap_is_full():
    h = heap.Heap()

    h.insert(0)

    status = h.is_full()

    assert status == True


def test_heap_is_empty():
    h = heap.Heap()

    status = h.is_empty()

    assert status == True


def test_heap_dad():
    h = heap.Heap(size=10)

    dad = h.dad(5)

    assert dad == 2


def test_heap_left_son():
    h = heap.Heap(size=10)

    left_son = h.left_son(5)

    assert left_son == 11


def test_heap_right_son():
    h = heap.Heap(size=10)

    right_son = h.right_son(5)

    assert right_son == 12


def test_heap_insert():
    h = heap.Heap()

    h.insert(0)

    status = h.insert(1)

    assert status == False


def test_heap_remove():
    h = heap.Heap()

    status = h.remove()

    assert status == False
