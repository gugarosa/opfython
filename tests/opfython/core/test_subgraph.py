import numpy as np
import pytest

from opfython.core import Subgraph
from opfython.utils import constants, exception


def test_subgraph_defaults():
    subgraph = Subgraph()

    assert subgraph.n_nodes == 0
    assert subgraph.n_features == 0
    assert subgraph.nodes == []
    assert subgraph.idx_nodes == []
    assert subgraph.trained is False


@pytest.mark.parametrize("suffix", ["csv", "json", "txt"])
def test_subgraph_loads_supported_files(suffix):
    X, Y = Subgraph()._load(f"data/boat.{suffix}")

    assert X.shape == (100, 2)
    assert Y.shape == (100,)


def test_subgraph_rejects_unknown_file_type():
    with pytest.raises(exception.ArgumentError):
        Subgraph()._load("data/boat.dat")


def test_subgraph_builds_with_indexes():
    X, Y = Subgraph()._load("data/boat.txt")
    indexes = np.arange(len(X)) + 100
    subgraph = Subgraph(X, Y, indexes)

    assert subgraph.n_nodes == 100
    assert subgraph.n_features == 2
    assert subgraph.nodes[0].idx == 100


def test_subgraph_resets_paths_and_arcs():
    subgraph = Subgraph(from_file="data/boat.txt")
    subgraph.nodes[0].adjacency = [1]
    subgraph.nodes[0].n_plateaus = 1
    subgraph.mark_nodes(0)

    assert subgraph.nodes[0].relevant == constants.RELEVANT

    subgraph.reset()

    assert subgraph.nodes[0].pred == constants.NIL
    assert subgraph.nodes[0].relevant == constants.IRRELEVANT
    assert subgraph.nodes[0].n_plateaus == 0
    assert subgraph.nodes[0].adjacency == []


@pytest.mark.parametrize(
    ("attribute", "value", "error"),
    [
        ("n_nodes", 1.5, exception.TypeError),
        ("n_nodes", -1, exception.ValueError),
        ("n_features", 1.5, exception.TypeError),
        ("n_features", -1, exception.ValueError),
        ("nodes", (), exception.TypeError),
        ("idx_nodes", (), exception.TypeError),
        ("trained", 1, exception.TypeError),
    ],
)
def test_subgraph_validates_public_attributes(attribute, value, error):
    subgraph = Subgraph()

    with pytest.raises(error):
        setattr(subgraph, attribute, value)
