# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np

from opfython.models.semi_supervised import SemiSupervisedOPF
from opfython.stream import loader, parser

X, Y = parser.parse_loader(loader.load_csv("data/boat.csv"))


def test_semi_supervised_fit():
    classifier = SemiSupervisedOPF()

    classifier.fit(X, Y, X)

    assert classifier.subgraph.trained is True


def test_semi_supervised_uses_precomputed_distances():
    classifier = SemiSupervisedOPF()
    classifier.pre_computed_distance = True
    classifier.pre_distances = np.ones((200, 200))

    classifier.fit(X, Y, X)

    assert classifier.subgraph.trained is True


def test_semi_supervised_fits_single_labeled_sample():
    classifier = SemiSupervisedOPF(distance="euclidean")

    classifier.fit(
        np.asarray([[0.0]]),
        np.asarray([0]),
        np.asarray([[1.0], [2.0]]),
    )

    assert classifier.predict(np.asarray([[-1.0], [4.0]])) == [0, 0]
    assert sorted(classifier.subgraph.idx_nodes) == [0, 1, 2]


def test_semi_supervised_indexed_distances_include_unlabeled_nodes(tmp_path):
    features = np.asarray([[0.0], [0.1], [10.0], [10.1], [0.2], [9.9]])
    labels = np.asarray([0, 0, 1, 1])
    path = tmp_path / "distances.txt"
    np.savetxt(path, np.abs(features - features.T))
    classifier = SemiSupervisedOPF(distance="euclidean", pre_computed_distance=path)

    classifier.fit(features[:4], labels, features[4:])

    assert classifier.predict(features[4:], np.asarray([4, 5])) == [0, 1]
    assert [node.predicted_label for node in classifier.subgraph.nodes[4:]] == [0, 1]


def test_semi_supervised_unlabeled_ids_do_not_follow_labeled_ids():
    features = np.asarray([[0.0], [0.1], [10.0], [10.1], [0.2], [9.9]])
    labels = np.asarray([0, 0, 1, 1])
    train_ids = np.asarray([9, 7, 8, 6])
    all_ids = np.concatenate((train_ids, [4, 5]))
    classifier = SemiSupervisedOPF(distance="euclidean")
    classifier.pre_computed_distance = True
    classifier.pre_distances = np.full((10, 10), 100.0)
    classifier.pre_distances[np.ix_(all_ids, all_ids)] = np.abs(features - features.T)
    direct = SemiSupervisedOPF(distance="euclidean")

    classifier.fit(features[:4], labels, features[4:], train_ids)
    direct.fit(features[:4], labels, features[4:], train_ids)

    assert [node.idx for node in classifier.subgraph.nodes] == all_ids.tolist()
    assert [node.label for node in classifier.subgraph.nodes] == [0, 0, 1, 1, 0, 1]
    assert classifier.subgraph.idx_nodes == direct.subgraph.idx_nodes
    assert [node.pred for node in classifier.subgraph.nodes] == [node.pred for node in direct.subgraph.nodes]
    assert classifier.predict(features[4:], np.asarray([4, 5])) == direct.predict(features[4:]) == [0, 1]
