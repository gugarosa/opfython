# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import pytest

from opfython.models.knn_supervised import KNNSupervisedOPF
from opfython.stream import loader, parser
from opfython.subgraphs.knn import KNNSubgraph
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
        classifier.fit(features, np.asarray([0, 0, 1, 1]), validation, np.asarray([0, 0]))

    assert classifier.subgraph.trained
    assert classifier.predict(validation) == [1, 0]


def test_knn_supervised_learning_does_not_call_public_prediction(monkeypatch):
    features = np.asarray([[0.0], [0.1], [10.0], [10.1]])
    labels = np.asarray([0, 0, 1, 1])
    classifier = KNNSupervisedOPF(max_k=3, distance="euclidean")
    public_calls = []
    monkeypatch.setattr(classifier, "predict", lambda *args: public_calls.append(args))

    classifier._learn(features, labels, None, features, labels, None)

    assert public_calls == []
    assert classifier.subgraph.trained is False
    assert classifier.subgraph.best_k == 1
    assert all(node.adjacency == [] for node in classifier.subgraph.nodes)


def test_knn_supervised_failed_validation_leaves_prediction_unavailable():
    classifier = KNNSupervisedOPF(distance="euclidean")
    classifier.pre_computed_distance = True
    classifier.pre_distances = np.ones((4, 4))
    features = np.asarray([[0.0], [0.1], [10.0], [10.1]])
    labels = np.asarray([0, 0, 1, 1])

    with pytest.raises(exception.BuildError):
        classifier.fit(features, labels, features[:1], labels[:1], I_val=np.asarray([4]))

    assert isinstance(classifier.subgraph, KNNSubgraph)
    assert classifier.subgraph.trained is False
    with pytest.raises(exception.BuildError, match="`subgraph.trained` is not True"):
        classifier.predict(features)
