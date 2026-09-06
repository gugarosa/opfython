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


@pytest.mark.parametrize("k", [0, 1, 4])
def test_knn_subgraph_retains_available_neighbours(k):
    subgraph = KNNSubgraph(np.arange(4.0).reshape(-1, 1))

    max_distances = subgraph.create_arcs(k, distance.euclidean_distance)

    assert max_distances.shape == (k,)
    assert all(len(node.adjacency) == min(k, 3) for node in subgraph.nodes)


@pytest.mark.parametrize("k", [1, np.int32(1), np.int64(1), np.asarray(1)])
def test_knn_subgraph_preserves_numpy_integer_indexes(k):
    subgraph = KNNSubgraph(np.arange(4.0).reshape(-1, 1))

    subgraph.create_arcs(k, distance.euclidean_distance)

    assert all(len(node.adjacency) == 1 for node in subgraph.nodes)
    assert k == 1


@pytest.mark.parametrize("k", [-1, 1.5])
def test_knn_subgraph_invalid_k_does_not_destroy_existing_arcs(k):
    subgraph = KNNSubgraph(np.arange(4.0).reshape(-1, 1))
    subgraph.create_arcs(1, distance.euclidean_distance)
    adjacency = [node.adjacency.copy() for node in subgraph.nodes]
    density = subgraph.density

    with pytest.raises((TypeError, ValueError)):
        subgraph.create_arcs(k, distance.euclidean_distance)

    assert [node.adjacency for node in subgraph.nodes] == adjacency
    assert subgraph.density == density


def test_knn_subgraph_rebuilding_arcs_replaces_old_state():
    features = np.asarray([[0.0], [1.0], [3.0], [10.0]])
    rebuilt = KNNSubgraph(features)
    fresh = KNNSubgraph(features)

    rebuilt.create_arcs(3, distance.euclidean_distance)
    rebuilt.create_arcs(1, distance.euclidean_distance)
    fresh.create_arcs(1, distance.euclidean_distance)

    assert [node.adjacency for node in rebuilt.nodes] == [
        node.adjacency for node in fresh.nodes
    ]
    assert rebuilt.density == fresh.density
    assert [node.radius for node in rebuilt.nodes] == [
        node.radius for node in fresh.nodes
    ]
