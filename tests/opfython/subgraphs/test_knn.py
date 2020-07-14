import numpy as np

from opfython.math import distance
from opfython.stream import loader, parser
from opfython.subgraphs import knn

csv = loader.load_csv('data/boat.csv')
X, Y = parser.parse_loader(csv)


def test_knn_subgraph_n_clusters():
    subgraph = knn.KNNSubgraph(X, Y)

    assert subgraph.n_clusters == 0


def test_knn_subgraph_n_clusters_setter():
    subgraph = knn.KNNSubgraph(X, Y)

    try:
        subgraph.n_clusters = 0.5
    except:
        subgraph.n_clusters = 1

    assert subgraph.n_clusters == 1

    try:
        subgraph.n_clusters = -1
    except:
        subgraph.n_clusters = 1

    assert subgraph.n_clusters == 1


def test_knn_subgraph_best_k():
    subgraph = knn.KNNSubgraph(X, Y)

    assert subgraph.best_k == 0


def test_knn_subgraph_best_k_setter():
    subgraph = knn.KNNSubgraph(X, Y)

    try:
        subgraph.best_k = 0.5
    except:
        subgraph.best_k = 1

    assert subgraph.best_k == 1
    
    try:
        subgraph.best_k = -1
    except:
        subgraph.best_k = 1

    assert subgraph.best_k == 1


def test_knn_subgraph_constant():
    subgraph = knn.KNNSubgraph(X, Y)

    assert subgraph.constant == 0.0


def test_knn_subgraph_constant_setter():
    subgraph = knn.KNNSubgraph(X, Y)

    try:
        subgraph.constant = 'a'
    except:
        subgraph.constant = 2.5

    assert subgraph.constant == 2.5


def test_knn_subgraph_density():
    subgraph = knn.KNNSubgraph(X, Y)

    assert subgraph.density == 0.0


def test_knn_subgraph_density_setter():
    subgraph = knn.KNNSubgraph(X, Y)

    try:
        subgraph.density = 'a'
    except:
        subgraph.density = 2.5

    assert subgraph.density == 2.5


def test_knn_subgraph_min_density():
    subgraph = knn.KNNSubgraph(X, Y)

    assert subgraph.min_density == 0.0


def test_knn_subgraph_min_density_setter():
    subgraph = knn.KNNSubgraph(X, Y)

    try:
        subgraph.min_density = 'a'
    except:
        subgraph.min_density = 2.5

    assert subgraph.min_density == 2.5


def test_knn_subgraph_max_density():
    subgraph = knn.KNNSubgraph(X, Y)

    assert subgraph.max_density == 0.0


def test_knn_subgraph_max_density_setter():
    subgraph = knn.KNNSubgraph(X, Y)

    try:
        subgraph.max_density = 'a'
    except:
        subgraph.max_density = 2.5

    assert subgraph.max_density == 2.5


def test_knn_subgraph_calculate_pdf():
    subgraph = knn.KNNSubgraph(X, Y)

    distances = np.ones((100, 100))

    subgraph.create_arcs(1, distance.euclidean_distance,
                         pre_computed_distance=True, pre_distances=distances)
    subgraph.calculate_pdf(1, distance.euclidean_distance,
                           pre_computed_distance=True, pre_distances=distances)

    subgraph.create_arcs(1, distance.euclidean_distance)
    subgraph.calculate_pdf(1, distance.euclidean_distance)

    assert subgraph.min_density != 0
    assert subgraph.max_density != 0


def test_knn_subgraph_create_arcs():
    subgraph = knn.KNNSubgraph(X, Y)

    distances = np.ones((100, 100))

    distances.fill(0.000001)

    subgraph.create_arcs(1, distance.euclidean_distance,
                         pre_computed_distance=True, pre_distances=distances)

    max_distances = subgraph.create_arcs(1, distance.euclidean_distance)

    assert len(max_distances) == 1


def test_knn_subgraph_eliminate_maxima_height():
    subgraph = knn.KNNSubgraph(X, Y)

    subgraph.eliminate_maxima_height(2.5)

    assert subgraph.nodes[0].cost == 0
