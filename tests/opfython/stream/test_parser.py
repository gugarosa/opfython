import numpy as np
import pytest

from opfython.stream import loader, parser
from opfython.utils import exception


def test_parse_loader():
    X, Y = parser.parse_loader(loader.load_csv("data/boat.csv"))

    assert X.shape == (100, 2)
    assert Y.shape == (100,)


def test_parse_loader_handles_non_array_input():
    assert parser.parse_loader([]) == (None, None)


def test_parse_loader_rejects_nonsequential_labels():
    data = np.asarray([[0, 0, 1], [1, 2, 1]])

    with pytest.raises(exception.ValueError):
        parser.parse_loader(data)
