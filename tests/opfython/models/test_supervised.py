import numpy as np

from opfython.models import supervised
from opfython.stream import loader, parser, splitter

csv = loader.load_csv('data/boat.csv')
X, Y = parser.parse_loader(csv)


def test_supervised_opf_fit():
    opf = supervised.SupervisedOPF()

    opf.fit(X, Y)

    assert opf.subgraph.trained == True

    opf.pre_computed_distance = True
    try:
        opf.pre_distances = np.ones((99, 99))
        opf.fit(X, Y)
    except:
        opf.pre_distances = np.ones((100, 100))
        opf.fit(X, Y)

    assert opf.subgraph.trained == True


def test_supervised_opf_predict():
    opf = supervised.SupervisedOPF()

    try:
        _ = opf.predict(X)
    except:
        opf.fit(X, Y)
        preds = opf.predict(X)

    assert len(preds) == 100

    try:
        opf.fit(X, Y)
        opf.subgraph.trained = False
        _ = opf.predict(X)
    except:
        opf.fit(X, Y)
        preds = opf.predict(X)

    assert len(preds) == 100

    opf.pre_computed_distance = True
    opf.pre_distances = np.ones((100, 100))

    opf.fit(X, Y)
    preds = opf.predict(X)

    assert len(preds) == 100


def test_supervised_opf_learn():
    opf = supervised.SupervisedOPF()

    X_train, X_val, Y_train, Y_val = splitter.split(
        X, Y, percentage=0.1, random_state=1)

    opf.learn(X_train, Y_train, X_val, Y_val, n_iterations=5)

    assert isinstance(opf, supervised.SupervisedOPF)


def test_supervised_opf_prune():
    opf = supervised.SupervisedOPF()

    X_train, X_val, Y_train, Y_val = splitter.split(
        X, Y, percentage=0.1, random_state=1)

    opf.prune(X_train, Y_train, X_val, Y_val, n_iterations=5)

    assert opf.subgraph.n_nodes == 10
