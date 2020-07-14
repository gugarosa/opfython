from opfython.core import subgraph
from opfython.utils import constants


def test_subgraph_n_nodes():
    s = subgraph.Subgraph()

    assert s.n_nodes == 0


def test_subgraph_n_nodes_setter():
    s = subgraph.Subgraph()

    try:
        s.n_nodes = 10.5
    except:
        s.n_nodes = 0

    assert s.n_nodes == 0

    try:
        s.n_nodes = -1
    except:
        s.n_nodes = 0

    assert s.n_nodes == 0


def test_subgraph_n_features():
    s = subgraph.Subgraph()

    assert s.n_features == 0


def test_subgraph_n_features_setter():
    s = subgraph.Subgraph()

    try:
        s.n_features = 10.5
    except:
        s.n_features = 1

    assert s.n_features == 1

    try:
        s.n_features = -1
    except:
        s.n_features = 1

    assert s.n_features == 1


def test_subgraph_nodes():
    s = subgraph.Subgraph()

    assert isinstance(s.nodes, list)


def test_subgraph_nodes_setter():
    s = subgraph.Subgraph()

    try:
        s.nodes = 10
    except:
        s.nodes = []

    assert isinstance(s.nodes, list)


def test_subgraph_idx_nodes():
    s = subgraph.Subgraph()

    assert isinstance(s.idx_nodes, list)


def test_subgraph_idx_nodes_setter():
    s = subgraph.Subgraph()

    try:
        s.idx_nodes = 10
    except:
        s.idx_nodes = []

    assert isinstance(s.idx_nodes, list)


def test_subgraph_trained():
    s = subgraph.Subgraph()

    assert s.trained == False


def test_subgraph_trained_setter():
    s = subgraph.Subgraph()

    try:
        s.trained = 10
    except:
        s.trained = True

    assert s.trained == True


def test_subgraph_load():
    s = subgraph.Subgraph()

    try:
        X, Y = s._load('data/boat')
    except:
        X, Y = s._load('data/boat.csv')
        X, Y = s._load('data/boat.json')
        X, Y = s._load('data/boat.txt')

    assert X.shape == (100, 2)
    assert Y.shape == (100,)


def test_subgraph_build():
    s = subgraph.Subgraph()

    X, Y = s._load('data/boat.txt')

    s._build(X, Y, None)

    assert len(s.nodes) == 100
    assert s.n_features == 2


def test_subgraph_build_with_index():
    s = subgraph.Subgraph()

    X, Y = s._load('data/boat.txt')

    I = Y

    s._build(X, Y, I)

    assert len(s.nodes) == 100
    assert s.n_features == 2


def test_subgraph_destroy_arcs():
    s = subgraph.Subgraph(from_file='data/boat.txt')

    s.destroy_arcs()

    assert s.nodes[0].n_plateaus == 0
    assert len(s.nodes[0].adjacency) == 0


def test_subgraph_mark_nodes():
    s = subgraph.Subgraph(from_file='data/boat.txt')

    s.mark_nodes(0)

    assert s.nodes[0].relevant == constants.RELEVANT


def test_subgraph_reset():
    s = subgraph.Subgraph(from_file='data/boat.txt')

    s.reset()

    assert s.nodes[0].pred == constants.NIL
    assert s.nodes[0].relevant == constants.IRRELEVANT
