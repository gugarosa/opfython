import sys

from opfython.utils import constants


def test_constants():
    assert constants.EPSILON == 1e-20

    assert constants.FLOAT_MAX == sys.float_info.max

    assert constants.WHITE == 0
    assert constants.GRAY == 1
    assert constants.BLACK == 2

    assert constants.NIL == -1

    assert constants.STANDARD == 0
    assert constants.PROTOTYPE == 1

    assert constants.IRRELEVANT == 0
    assert constants.RELEVANT == 1

    assert constants.MAX_ARC_WEIGHT == 100000

    assert constants.MAX_DENSITY == 1000
