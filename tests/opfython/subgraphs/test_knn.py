import numpy as np
import pytest

from opfython.math import distance
from opfython.stream import loader, parser
from opfython.subgraphs import KNNSubgraph
from opfython.utils import exception

X, Y = parser.parse_loader(loader.load_csv("data/boat.csv"))


def test_knn_subgraph_defaults():
    subgraph = KNNSubgraph(X, Y)

    assert subgraph.n_clusters == 0
    assert subgraph.best_k == 0
    assert subgraph.constant == 0
    assert subgraph.density == 0
    assert subgraph.min_density == 0
    assert subgraph.max_density == 0


def test_knn_subgraph_calculates_pdf():
    subgraph = KNNSubgraph(X, Y)
    distances = np.ones((100, 100))

    subgraph.create_arcs(
        1,
        distance.euclidean_distance,
        pre_computed_distance=True,
        pre_distances=distances,
    )
    subgraph.calculate_pdf(
        1,
        distance.euclidean_distance,
        pre_computed_distance=True,
        pre_distances=distances,
    )

    assert subgraph.min_density != 0
    assert subgraph.max_density != 0


def test_knn_subgraph_creates_arcs_from_features():
    subgraph = KNNSubgraph(X, Y)

    max_distances = subgraph.create_arcs(1, distance.euclidean_distance)

    assert len(max_distances) == 1
    assert len(subgraph.nodes[0].adjacency) == 1


def test_knn_subgraph_eliminates_maxima():
    subgraph = KNNSubgraph(X, Y)

    subgraph.eliminate_maxima_height(2.5)

    assert subgraph.nodes[0].cost == 0


@pytest.mark.parametrize(
    ("attribute", "value", "error"),
    [
        ("n_clusters", 0.5, exception.TypeError),
        ("n_clusters", -1, exception.ValueError),
        ("best_k", 0.5, exception.TypeError),
        ("best_k", -1, exception.ValueError),
        ("constant", "invalid", exception.TypeError),
        ("density", "invalid", exception.TypeError),
        ("min_density", "invalid", exception.TypeError),
        ("max_density", "invalid", exception.TypeError),
    ],
)
def test_knn_subgraph_validates_public_attributes(attribute, value, error):
    subgraph = KNNSubgraph(X, Y)

    with pytest.raises(error):
        setattr(subgraph, attribute, value)
