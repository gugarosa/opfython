# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import pytest

from opfython.math import distance
from opfython.stream import loader, parser
from opfython.subgraphs.knn import KNNSubgraph
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

    assert [node.adjacency for node in rebuilt.nodes] == [node.adjacency for node in fresh.nodes]
    assert rebuilt.density == fresh.density
    assert [node.radius for node in rebuilt.nodes] == [node.radius for node in fresh.nodes]


@pytest.mark.parametrize("k", [0, 1, 3, 6])
def test_knn_subgraph_indexed_arcs_match_direct_ties_and_density_state(k):
    features = np.asarray([[-1.0], [1.0], [0.0], [0.0]])
    indexes = np.asarray([3, 1, 0, 2])
    matrix = np.empty((4, 4))
    matrix[np.ix_(indexes, indexes)] = np.abs(features - features.T)
    matrix.flags.writeable = False
    direct = KNNSubgraph(features, I=indexes)
    precomputed = KNNSubgraph(features, I=indexes)

    direct_maxima = direct.create_arcs(k, distance.euclidean_distance)
    precomputed_maxima = precomputed.create_arcs(k, distance.euclidean_distance, True, matrix)

    np.testing.assert_array_equal(precomputed_maxima, direct_maxima)
    assert precomputed.density == direct.density
    for index, (actual, expected) in enumerate(zip(precomputed.nodes, direct.nodes)):
        neighbours = sorted(
            (other for other in range(len(features)) if other != index),
            key=lambda other: abs(features[index, 0] - features[other, 0]),
        )[:k]
        assert actual.adjacency == expected.adjacency == neighbours
        assert actual.radius == expected.radius
        assert actual.n_plateaus == expected.n_plateaus == 0
    if 0 < k < len(features):
        direct.calculate_pdf(k, distance.euclidean_distance)
        precomputed.calculate_pdf(k, distance.euclidean_distance, True, matrix)
        assert precomputed.min_density == direct.min_density
        assert precomputed.max_density == direct.max_density
        assert precomputed.constant == direct.constant
        assert [node.density for node in precomputed.nodes] == [node.density for node in direct.nodes]
        assert [node.cost for node in precomputed.nodes] == [node.cost for node in direct.nodes]
