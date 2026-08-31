import numpy as np
import pytest

from opfython.models import KNNSupervisedOPF
from opfython.stream import loader, parser
from opfython.utils import exception

X, Y = parser.parse_loader(loader.load_csv("data/boat.csv"))


def test_knn_supervised_rejects_invalid_max_k():
    with pytest.raises(exception.TypeError):
        KNNSupervisedOPF(max_k=1.5)
    with pytest.raises(exception.ValueError):
        KNNSupervisedOPF(max_k=0)

    classifier = KNNSupervisedOPF()
    with pytest.raises(exception.ValueError):
        classifier.max_k = 0


def test_knn_supervised_fit_and_predict():
    classifier = KNNSupervisedOPF()

    classifier.fit(X, Y, X, Y)
    predictions = classifier.predict(X)

    assert classifier.subgraph.trained is True
    assert len(predictions) == 100


def test_knn_supervised_validates_precomputed_distances():
    classifier = KNNSupervisedOPF()
    classifier.pre_computed_distance = True
    classifier.pre_distances = np.ones((99, 99))

    with pytest.raises(exception.BuildError):
        classifier.fit(X, Y, X, Y)

    classifier.pre_distances = np.ones((100, 100))
    classifier.fit(X, Y, X, Y)

    assert len(classifier.predict(X)) == 100
