import numpy as np

from opfython.core import node
from opfython.utils import constants


def test_node_idx():
    n = node.Node()

    assert n.idx == 0


def test_node_idx_setter():
    n = node.Node()

    try:
        n.idx = 10.5
    except:
        n.idx = 0

    assert n.idx == 0

    try:
        n.idx = -1
    except:
        n.idx = 0

    assert n.idx == 0


def test_node_label():
    n = node.Node()

    assert n.label == 0


def test_node_label_setter():
    n = node.Node()

    try:
        n.label = 10.5
    except:
        n.label = 1

    assert n.label == 1

    try:
        n.label = -1
    except:
        n.label = 1

    assert n.label == 1


def test_node_predicted_label():
    n = node.Node()

    assert n.predicted_label == 0


def test_node_predicted_label_setter():
    n = node.Node()

    try:
        n.predicted_label = 10.5
    except:
        n.predicted_label = 0

    assert n.predicted_label == 0

    try:
        n.predicted_label = -1
    except:
        n.predicted_label = 0

    assert n.predicted_label == 0


def test_node_cluster_label():
    n = node.Node()

    assert n.cluster_label == 0


def test_node_cluster_label_setter():
    n = node.Node()

    try:
        n.cluster_label = 10.5
    except:
        n.cluster_label = 0

    assert n.cluster_label == 0

    try:
        n.cluster_label = -1
    except:
        n.cluster_label = 0

    assert n.cluster_label == 0


def test_node_features():
    n = node.Node()

    assert isinstance(n.features, np.ndarray)


def test_node_features_setter():
    n = node.Node()

    try:
        n.features = []
    except:
        n.features = np.asarray([])

    assert isinstance(n.features, np.ndarray)


def test_node_cost():
    n = node.Node()

    assert n.cost == 0


def test_node_cost_setter():
    n = node.Node()

    try:
        n.cost = 'a'
    except:
        n.cost = 1.5

    assert n.cost == 1.5


def test_node_density():
    n = node.Node()

    assert n.density == 0


def test_node_density_setter():
    n = node.Node()

    try:
        n.density = 'a'
    except:
        n.density = 2.25

    assert n.density == 2.25


def test_node_radius():
    n = node.Node()

    assert n.radius == 0


def test_node_radius_setter():
    n = node.Node()

    try:
        n.radius = 'a'
    except:
        n.radius = 0.5

    assert n.radius == 0.5


def test_node_n_plateaus():
    n = node.Node()

    assert n.n_plateaus == 0


def test_node_n_plateaus_setter():
    n = node.Node()

    try:
        n.n_plateaus = 10.5
    except:
        n.n_plateaus = 0

    assert n.n_plateaus == 0

    try:
        n.n_plateaus = -1
    except:
        n.n_plateaus = 0

    assert n.n_plateaus == 0


def test_node_adjacency():
    n = node.Node()

    assert isinstance(n.adjacency, list)


def test_node_adjacency_setter():
    n = node.Node()

    try:
        n.adjacency = ''
    except:
        n.adjacency = []

    assert isinstance(n.adjacency, list)


def test_node_root():
    n = node.Node()

    assert n.root == 0


def test_node_root_setter():
    n = node.Node()

    try:
        n.root = 10.5
    except:
        n.root = 0

    assert n.root == 0

    try:
        n.root = -1
    except:
        n.root = 0

    assert n.root == 0


def test_node_status():
    n = node.Node()

    assert n.status == constants.STANDARD


def test_node_status_setter():
    n = node.Node()

    try:
        n.status = 'a'
    except:
        n.status = constants.PROTOTYPE

    assert n.status == constants.PROTOTYPE


def test_node_pred():
    n = node.Node()

    assert n.pred == constants.NIL


def test_node_pred_setter():
    n = node.Node()

    try:
        n.pred = 'a'
    except:
        n.pred = constants.NIL

    assert n.pred == constants.NIL

    try:
        n.pred = -2
    except:
        n.pred = constants.NIL

    assert n.pred == constants.NIL


def test_node_relevant():
    n = node.Node()

    assert n.relevant == constants.IRRELEVANT


def test_node_relevant_setter():
    n = node.Node()

    try:
        n.relevant = 'a'
    except:
        n.relevant = constants.RELEVANT

    assert n.relevant == constants.RELEVANT
