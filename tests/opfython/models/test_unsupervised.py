# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import pytest

from opfython.models.unsupervised import UnsupervisedOPF
from opfython.stream import loader, parser
from opfython.subgraphs.knn import KNNSubgraph
from opfython.utils import constants, exception

X, Y = parser.parse_loader(loader.load_csv("data/boat.csv"))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"min_k": 1.5}, exception.TypeError),
        ({"max_k": 1.5}, exception.TypeError),
        ({"min_k": 0}, exception.ValueError),
        ({"min_k": 2, "max_k": 1}, exception.ValueError),
    ],
)
def test_unsupervised_rejects_invalid_k(kwargs, error):
    with pytest.raises(error):
        UnsupervisedOPF(**kwargs)


def test_unsupervised_requires_fitted_subgraph():
    classifier = UnsupervisedOPF()

    with pytest.raises(exception.BuildError):
        classifier.predict(X)


def test_unsupervised_validates_public_k_values():
    classifier = UnsupervisedOPF()

    with pytest.raises(exception.TypeError):
        classifier.min_k = 1.5
    with pytest.raises(exception.ValueError):
        classifier.min_k = 0
    classifier.min_k = 2
    with pytest.raises(exception.ValueError):
        classifier.max_k = 1


def test_unsupervised_checks_mutated_k_range_before_fitting():
    classifier = UnsupervisedOPF()
    classifier.min_k = 2

    with pytest.raises(exception.ValueError):
        classifier.fit(np.arange(4.0).reshape(-1, 1))


def test_unsupervised_fit_predict_and_propagate():
    classifier = UnsupervisedOPF()

    classifier.fit(X, Y)
    predictions, clusters = classifier.predict(X)
    classifier.propagate_labels()

    assert classifier.subgraph.trained is True
    assert len(predictions) == 100
    assert len(clusters) == 100
    assert classifier.subgraph.nodes[0].predicted_label == 0


def test_unsupervised_uses_precomputed_distances():
    classifier = UnsupervisedOPF()
    classifier.pre_computed_distance = True
    classifier.pre_distances = np.ones((100, 100))

    classifier.fit(X, Y)
    predictions, clusters = classifier.predict(X)

    assert len(predictions) == 100
    assert len(clusters) == 100


def test_unsupervised_selected_forest_matches_fixed_k():
    features = np.random.default_rng(1).normal(size=(8, 2))
    searched = UnsupervisedOPF(min_k=2, max_k=4, distance="euclidean")
    searched.fit(features)
    fixed = UnsupervisedOPF(
        min_k=searched.subgraph.best_k,
        max_k=searched.subgraph.best_k,
        distance="euclidean",
    )
    fixed.fit(features)

    assert searched.subgraph.density == fixed.subgraph.density
    np.testing.assert_allclose(
        [node.cost for node in searched.subgraph.nodes],
        [node.cost for node in fixed.subgraph.nodes],
    )
    assert [node.root for node in searched.subgraph.nodes] == [node.root for node in fixed.subgraph.nodes]
    assert sorted(searched.subgraph.idx_nodes) == list(range(len(features)))


def test_unsupervised_plateau_arcs_are_unique_and_symmetric():
    classifier = UnsupervisedOPF(min_k=2, max_k=2, distance="euclidean")
    classifier.subgraph = KNNSubgraph(np.zeros((4, 2)))
    classifier.subgraph.create_arcs(2, classifier.distance_fn)
    classifier.subgraph.calculate_pdf(2, classifier.distance_fn)

    classifier._clustering(2)

    for index, node in enumerate(classifier.subgraph.nodes):
        assert len(node.adjacency) == len(set(node.adjacency))
        assert index not in node.adjacency
        for adjacent in node.adjacency:
            assert index in classifier.subgraph.nodes[int(adjacent)].adjacency
    assert classifier.subgraph.n_clusters == 1


def test_unsupervised_clusters_identical_samples_without_invalid_densities():
    classifier = UnsupervisedOPF(min_k=2, max_k=3, distance="euclidean")

    with np.errstate(divide="raise", invalid="raise"):
        classifier.fit(np.zeros((4, 2)))

    assert classifier.subgraph.n_clusters == 1
    assert all(node.density == constants.MAX_DENSITY for node in classifier.subgraph.nodes)
    assert classifier.predict(np.zeros((2, 2))) == ([0, 0], [0, 0])


def test_unsupervised_candidate_cuts_match_independent_forests(monkeypatch):
    features = np.random.default_rng(1).normal(size=(8, 2))
    classifier = UnsupervisedOPF(min_k=2, max_k=4, distance="euclidean")
    normalized_cut = classifier._normalized_cut
    candidate_cuts = {}

    def record_cut(k):
        cut = normalized_cut(k)
        candidate_cuts[k] = cut
        return cut

    monkeypatch.setattr(classifier, "_normalized_cut", record_cut)
    classifier.fit(features)

    assert len(candidate_cuts) > 1
    for k, cut in candidate_cuts.items():
        independent = UnsupervisedOPF(min_k=k, max_k=k, distance="euclidean")
        independent.fit(features)
        assert cut == pytest.approx(independent._normalized_cut(k))
    assert classifier.subgraph.best_k == min(candidate_cuts, key=candidate_cuts.get)
