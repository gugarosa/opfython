from opfython.utils import decorator


def test_avoid_zero_division():
    @decorator.avoid_zero_division
    def call(x, y):
        return x, y

    x, y = call(1, 1)

    assert x == 1
    assert y == 1
