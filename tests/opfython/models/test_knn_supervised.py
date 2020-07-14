import numpy as np

from opfython.models import knn_supervised
from opfython.stream import loader, parser, splitter

csv = loader.load_csv('data/boat.csv')
X, Y = parser.parse_loader(csv)


def test_knn_supervised_opf_max_k():
    opf = knn_supervised.KNNSupervisedOPF()

    assert opf.max_k == 1


def test_knn_supervised_opf_max_k_setter():
    opf = knn_supervised.KNNSupervisedOPF()

    try:
        opf.max_k = 1.5
    except:
        opf.max_k = 3

    assert opf.max_k == 3

    try:
        opf.max_k = 0
    except:
        opf.max_k = 3

    assert opf.max_k == 3


def test_knn_supervised_opf_fit():
    opf = knn_supervised.KNNSupervisedOPF()

    opf.fit(X, Y, X, Y)

    assert opf.subgraph.trained == True

    opf.pre_computed_distance = True
    try:
        opf.pre_distances = np.ones((99, 99))
        opf.fit(X, Y, X, Y)
    except:
        opf.pre_distances = np.ones((100, 100))
        opf.fit(X, Y, X, Y)

    assert opf.subgraph.trained == True


def test_knn_supervised_opf_predict():
    opf = knn_supervised.KNNSupervisedOPF()

    try:
        _ = opf.predict(X)
    except:
        opf.fit(X, Y, X, Y)
        preds = opf.predict(X)

    assert len(preds) == 100

    opf.pre_computed_distance = True
    opf.pre_distances = np.ones((100, 100))

    opf.fit(X, Y, X, Y)
    preds = opf.predict(X)

    assert len(preds) == 100
