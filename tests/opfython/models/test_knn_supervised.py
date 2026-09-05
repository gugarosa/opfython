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


def test_knn_supervised_selected_forest_matches_fixed_k():
    features = np.asarray([[0.0], [0.1], [10.0], [10.1]])
    labels = np.asarray([0, 0, 1, 1])
    searched = KNNSupervisedOPF(max_k=3, distance="euclidean")
    searched.fit(features, labels, features, labels)
    fixed = KNNSupervisedOPF(max_k=searched.subgraph.best_k, distance="euclidean")
    fixed.fit(features, labels, features, labels)

    assert searched.subgraph.density == fixed.subgraph.density
    np.testing.assert_allclose(
        [node.cost for node in searched.subgraph.nodes],
        [node.cost for node in fixed.subgraph.nodes],
    )
    assert sorted(searched.subgraph.idx_nodes) == list(range(len(features)))


def test_knn_supervised_handles_missing_classes_in_validation():
    features = np.asarray([[0.0], [0.1], [10.0], [10.1]])
    validation = np.asarray([[9.95], [0.05]])
    classifier = KNNSupervisedOPF(distance="euclidean")

    with np.errstate(divide="raise", invalid="raise"):
        classifier.fit(
            features, np.asarray([0, 0, 1, 1]), validation, np.asarray([0, 0])
        )

    assert classifier.subgraph.trained
    assert classifier.predict(validation) == [1, 0]
