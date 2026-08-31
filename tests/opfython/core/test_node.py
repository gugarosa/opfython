import numpy as np
import pytest

from opfython.core import Node
from opfython.utils import constants, exception


def test_node_defaults():
    node = Node()

    assert node.idx == 0
    assert node.label == 0
    assert node.predicted_label == 0
    assert node.cluster_label == 0
    assert isinstance(node.features, np.ndarray)
    assert node.cost == 0
    assert node.density == 0
    assert node.radius == 0
    assert node.n_plateaus == 0
    assert node.adjacency == []
    assert node.root == 0
    assert node.status == constants.STANDARD
    assert node.pred == constants.NIL
    assert node.relevant == constants.IRRELEVANT


def test_node_accepts_initial_values():
    features = np.asarray([1.0, 2.0])
    node = Node(2, 1, features)

    assert node.idx == 2
    assert node.label == 1
    assert np.array_equal(node.features, features)


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ((1.5,), exception.TypeError),
        ((-1,), exception.ValueError),
        ((0, 1.5), exception.TypeError),
        ((0, -1), exception.ValueError),
    ],
)
def test_node_rejects_invalid_identity(args, error):
    with pytest.raises(error):
        Node(*args)


@pytest.mark.parametrize(
    ("attribute", "value", "error"),
    [
        ("idx", 1.5, exception.TypeError),
        ("idx", -1, exception.ValueError),
        ("label", 1.5, exception.TypeError),
        ("label", -1, exception.ValueError),
        ("predicted_label", 1.5, exception.TypeError),
        ("cluster_label", -1, exception.ValueError),
        ("features", [], exception.TypeError),
        ("cost", "invalid", exception.TypeError),
        ("density", "invalid", exception.TypeError),
        ("radius", "invalid", exception.TypeError),
        ("n_plateaus", -1, exception.ValueError),
        ("adjacency", (), exception.TypeError),
        ("root", -1, exception.ValueError),
        ("status", -1, exception.TypeError),
        ("status", [], exception.TypeError),
        ("pred", -2, exception.ValueError),
        ("relevant", -1, exception.TypeError),
        ("relevant", [], exception.TypeError),
    ],
)
def test_node_validates_public_attributes(attribute, value, error):
    node = Node()

    with pytest.raises(error):
        setattr(node, attribute, value)
