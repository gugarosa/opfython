import numpy as np

from opfython.models import unsupervised
from opfython.stream import loader, parser

csv = loader.load_csv('data/boat.csv')
X, Y = parser.parse_loader(csv)


def test_unsupervised_opf_min_k():
    opf = unsupervised.UnsupervisedOPF()

    assert opf.min_k == 1


def test_unsupervised_opf_min_k_setter():
    opf = unsupervised.UnsupervisedOPF()

    try:
        opf.min_k = 1.5
    except:
        opf.min_k = 1

    assert opf.min_k == 1

    try:
        opf.min_k = 0
    except:
        opf.min_k = 1

    assert opf.min_k == 1


def test_unsupervised_opf_max_k():
    opf = unsupervised.UnsupervisedOPF()

    assert opf.max_k == 1


def test_unsupervised_opf_max_k_setter():
    opf = unsupervised.UnsupervisedOPF()
    opf.min_k = 2

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

    try:
        opf.max_k = 1
    except:
        opf.max_k = 3

    assert opf.max_k == 3


def test_unsupervised_opf_fit():
    opf = unsupervised.UnsupervisedOPF()

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


def test_unsupervised_opf_predict():
    opf = unsupervised.UnsupervisedOPF()

    try:
        _ = opf.predict(X)
    except:
        opf.fit(X, Y)
        preds, clusters = opf.predict(X)

    assert len(preds) == 100
    assert len(clusters) == 100

    try:
        opf.fit(X, Y)
        opf.subgraph.trained = False
        _, _ = opf.predict(X)
    except:
        opf.fit(X, Y)
        preds, clusters = opf.predict(X)

    assert len(preds) == 100
    assert len(clusters) == 100

    opf.pre_computed_distance = True
    opf.pre_distances = np.ones((100, 100))

    opf.fit(X, Y)
    preds, clusters = opf.predict(X)

    assert len(preds) == 100
    assert len(clusters) == 100


def test_unsupervised_opf_propagate_labels():
    opf = unsupervised.UnsupervisedOPF()

    opf.fit(X, Y)

    opf.propagate_labels()

    assert opf.subgraph.nodes[0].predicted_label == 0
