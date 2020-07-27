from opfython.utils import decorator


def test_avoid_zeros():
    @decorator.avoid_zeros
    def call(x, y):
        return x, y

    x, y = call(1, 1)

    assert x == 1
    assert y == 1
