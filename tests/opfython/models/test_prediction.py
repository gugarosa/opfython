# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import copy

import numpy as np
import pytest

from opfython.core.subgraph import Subgraph
from opfython.models.knn_supervised import KNNSupervisedOPF
from opfython.models.semi_supervised import SemiSupervisedOPF
from opfython.models.supervised import SupervisedOPF
from opfython.models.unsupervised import UnsupervisedOPF
from opfython.subgraphs.knn import KNNSubgraph
from opfython.utils import constants, exception

MODELS = [SupervisedOPF, SemiSupervisedOPF, KNNSupervisedOPF, UnsupervisedOPF]


def _fit(classifier, features, labels, indexes=None):
    if isinstance(classifier, KNNSupervisedOPF):
        classifier.fit(features, labels, features, labels, I_train=indexes, I_val=indexes)
    elif isinstance(classifier, SemiSupervisedOPF):
        classifier.fit(features, labels, np.empty((0, features.shape[1])), I_train=indexes)
    else:
        classifier.fit(features, labels, I_train=indexes)
    if isinstance(classifier, UnsupervisedOPF):
        classifier.propagate_labels()


@pytest.mark.parametrize("model_type", [KNNSupervisedOPF, UnsupervisedOPF])
@pytest.mark.parametrize("precomputed", [False, True])
@pytest.mark.parametrize(
    ("queries", "expected"),
    [([5.03, 5.03], [0, 0]), ([0.05, 9.95, 5.03], [0, 1, 0])],
)
def test_knn_predictions_do_not_depend_on_query_batch(model_type, precomputed, queries, expected, tmp_path):
    features = np.asarray([0.0, 0.1, 10.0, 10.1, *queries]).reshape(-1, 1)
    labels = np.asarray([0, 0, 1, 1])
    query_ids = np.arange(4, len(features))
    path = tmp_path / "distances.txt"
    np.savetxt(path, np.abs(features - features.T))
    classifier = model_type(
        distance="euclidean",
        pre_computed_distance=path if precomputed else None,
    )
    _fit(classifier, features[:4], labels, np.arange(4))

    batch = classifier.predict(features[query_ids], query_ids)
    singletons = [classifier.predict(features[index : index + 1], np.asarray([index])) for index in query_ids]
    reversed_batch = classifier.predict(features[query_ids[::-1]], query_ids[::-1])

    if model_type is UnsupervisedOPF:
        assert batch[0] == [result[0][0] for result in singletons] == expected
        assert batch[1] == [result[1][0] for result in singletons]
        assert reversed_batch == (batch[0][::-1], batch[1][::-1])
    else:
        assert batch == [result[0] for result in singletons] == expected
        assert reversed_batch == batch[::-1]


@pytest.mark.parametrize("model_type", MODELS)
def test_precomputed_distances_match_direct_predictions_for_sample_ids(model_type, tmp_path):
    features = np.asarray([[0.0], [10.0], [5.03], [0.07], [10.1], [0.1], [9.97]])
    train_ids = np.asarray([5, 1, 4, 0])
    test_ids = np.asarray([3, 6, 2])
    labels = np.asarray([0, 1, 1, 0])
    path = tmp_path / "distances.txt"
    np.savetxt(path, np.abs(features - features.T))
    direct = model_type(distance="euclidean")
    precomputed = model_type(distance="euclidean", pre_computed_distance=path)

    for classifier in (direct, precomputed):
        _fit(classifier, features[train_ids], labels, train_ids)

    assert precomputed.predict(features[test_ids], test_ids) == direct.predict(features[test_ids], test_ids)


@pytest.mark.parametrize("model_type", MODELS)
@pytest.mark.parametrize("shape", [(3, 3), (6, 3), (3,), ()])
def test_models_reject_distance_matrices_that_do_not_cover_training_ids(model_type, shape):
    classifier = model_type(distance="euclidean")
    classifier.pre_computed_distance = True
    classifier.pre_distances = np.ones(shape)

    with pytest.raises(exception.BuildError):
        _fit(
            classifier,
            np.asarray([[0.0], [0.1], [10.0], [10.1]]),
            np.asarray([0, 0, 1, 1]),
        )


@pytest.mark.parametrize("model_type", MODELS)
def test_models_reject_distance_matrices_that_do_not_cover_prediction_ids(
    model_type,
):
    features = np.asarray([[0.0], [0.1], [10.0], [10.1]])
    classifier = model_type(distance="euclidean")
    classifier.pre_computed_distance = True
    classifier.pre_distances = np.abs(features - features.T)
    _fit(classifier, features, np.asarray([0, 0, 1, 1]))

    with pytest.raises(exception.BuildError):
        classifier.predict(np.asarray([[0.05]]), np.asarray([4]))


@pytest.mark.parametrize("model_type", MODELS)
def test_precomputed_validation_preserves_supported_rectangular_matrices(
    model_type,
):
    features = np.asarray([[0.0], [0.1], [10.0], [10.1], [0.05], [9.95]])
    labels = np.asarray([0, 0, 1, 1])
    distances = np.abs(features - features.T)
    direct = model_type(distance="euclidean")
    precomputed = model_type(distance="euclidean")
    precomputed.pre_computed_distance = True
    if model_type in (KNNSupervisedOPF, UnsupervisedOPF):
        precomputed.pre_distances = distances[:, :4]
    else:
        precomputed.pre_distances = distances[:4, :]

    for classifier in (direct, precomputed):
        _fit(classifier, features[:4], labels)

    assert precomputed.predict(features[4:], np.asarray([4, 5])) == direct.predict(features[4:])


@pytest.mark.parametrize("model_type", [KNNSupervisedOPF, UnsupervisedOPF])
@pytest.mark.parametrize("n_samples", [0, 1, 4])
def test_knn_models_reject_unavailable_neighbourhoods(model_type, n_samples):
    classifier = model_type(max_k=max(1, n_samples))

    with pytest.raises(exception.ValueError):
        _fit(
            classifier,
            np.zeros((n_samples, 2)),
            np.zeros(n_samples, dtype=int),
        )


def _forest_state(classifier):
    state = copy.deepcopy(vars(classifier.subgraph))
    state["_nodes"] = [
        {name: value.tolist() if isinstance(value, np.ndarray) else value for name, value in vars(node).items()}
        for node in classifier.subgraph.nodes
    ]
    return state


@pytest.mark.parametrize("model_type", MODELS)
@pytest.mark.parametrize("graph_exists", [False, True])
def test_prediction_requires_a_completed_fit(model_type, graph_exists):
    classifier = model_type()
    if graph_exists:
        graph_type = KNNSubgraph if model_type in (KNNSupervisedOPF, UnsupervisedOPF) else Subgraph
        classifier.subgraph = graph_type(np.asarray([[0.0], [1.0]]))

    with pytest.raises(exception.BuildError, match="`subgraph(?:.trained)?`"):
        classifier.predict(np.asarray([[0.5]]))


@pytest.mark.parametrize("precomputed", [False, True])
@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        (
            SupervisedOPF,
            {
                "order": [2, 3, 4, 5, 1, 0],
                "cost": [3.0, 3.0, 0.0, 0.0, 1.0, 2.0],
                "pred": [1, 2, -1, -1, 3, 4],
                "root": [0, 0, 0, 0, 0, 0],
                "label": [0, 0, 0, 1, 1, 1],
                "cluster_label": [0, 0, 0, 0, 0, 0],
                "status": [0, 0, 1, 1, 0, 0],
            },
        ),
        (
            SemiSupervisedOPF,
            {
                "order": [2, 3, 4, 7, 5, 6, 0, 1],
                "cost": [2.0, 2.0, 0.0, 0.0, 1.0, 1.0, 2.0, 1.0],
                "pred": [6, 6, -1, -1, 3, 7, 2, 4],
                "root": [0, 0, 0, 0, 0, 0, 0, 0],
                "label": [0, 0, 0, 1, 1, 1, 0, 1],
                "cluster_label": [0, 0, 0, 0, 0, 0, 0, 0],
                "status": [0, 0, 1, 1, 0, 0, 0, 0],
            },
        ),
        (
            KNNSupervisedOPF,
            {
                "order": [0, 1, 4, 3, 5, 2],
                "cost": [1000.0, 1000.0, 1.0, 1000.0, 1000.0, 183.24309828254997],
                "pred": [-1, 0, -1, 4, -1, -1],
                "root": [0, 0, 2, 4, 4, 5],
                "label": [0, 0, 0, 1, 1, 1],
                "cluster_label": [0, 0, 0, 0, 0, 0],
                "status": [0, 0, 0, 0, 0, 0],
            },
        ),
        (
            UnsupervisedOPF,
            {
                "order": [0, 1, 4, 3, 5, 2],
                "cost": [1000.0, 1000.0, 1.0, 1000.0, 1000.0, 183.24309828254997],
                "pred": [-1, 0, -1, 4, -1, -1],
                "root": [0, 0, 2, 4, 4, 5],
                "label": [0, 0, 0, 1, 1, 1],
                "cluster_label": [0, 0, 3, 1, 1, 2],
                "status": [0, 0, 0, 0, 0, 0],
            },
        ),
    ],
)
def test_fitted_forests_retain_recorded_baseline_state(model_type, expected, precomputed):
    features = np.asarray([[0.0], [1.0], [4.0], [7.0], [8.0], [10.0], [2.0], [9.0], [0.5], [5.5], [9.0]])
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    query_ids = np.asarray([8, 9, 10])
    classifier = model_type(distance="euclidean")
    if precomputed:
        classifier.pre_computed_distance = True
        classifier.pre_distances = np.abs(features - features.T)

    if model_type is SemiSupervisedOPF:
        result = classifier.fit(features[:6], labels, features[6:8])
    elif model_type is KNNSupervisedOPF:
        result = classifier.fit(features[:6], labels, features[query_ids], np.asarray([0, 0, 1]), I_val=query_ids)
    else:
        result = classifier.fit(features[:6], labels)
    if model_type is UnsupervisedOPF:
        classifier.propagate_labels()

    assert result is None
    assert classifier.subgraph.trained is True
    assert classifier.subgraph.idx_nodes == expected["order"]
    assert [node.cost for node in classifier.subgraph.nodes] == pytest.approx(expected["cost"])
    for name in ("pred", "root", "label", "cluster_label", "status"):
        assert [getattr(node, name) for node in classifier.subgraph.nodes] == expected[name]
    assert [node.predicted_label for node in classifier.subgraph.nodes] == expected["label"]
    assert [node.idx for node in classifier.subgraph.nodes] == list(range(len(expected["order"])))
    predictions = classifier.predict(features[query_ids], query_ids)
    assert predictions == (([0, 0, 1], [0, 3, 1]) if model_type is UnsupervisedOPF else [0, 0, 1])


@pytest.mark.parametrize("model_type", MODELS)
@pytest.mark.parametrize("precomputed", [False, True])
def test_fitted_model_save_load_preserves_all_forest_state(model_type, precomputed, tmp_path):
    features = np.asarray([[0.0], [1.0], [4.0], [7.0], [8.0], [10.0], [0.5], [5.5], [9.0]])
    classifier = model_type(distance="euclidean")
    if precomputed:
        classifier.pre_computed_distance = True
        classifier.pre_distances = np.abs(features - features.T)
    _fit(classifier, features[:6], np.asarray([0, 0, 0, 1, 1, 1]))
    query_ids = np.asarray([6, 7, 8])
    predictions = classifier.predict(features[query_ids], query_ids)
    state = _forest_state(classifier)
    model_file = tmp_path / "fitted.pkl"

    assert classifier.save(model_file) is None
    restored = model_type()
    assert restored.load(model_file) is None

    assert type(restored) is model_type
    assert vars(restored).keys() == vars(classifier).keys()
    assert restored.distance == classifier.distance
    assert restored.distance_fn is classifier.distance_fn
    assert restored.pre_computed_distance is precomputed
    np.testing.assert_equal(restored.pre_distances, classifier.pre_distances)
    assert _forest_state(restored) == state
    assert restored.predict(features[query_ids], query_ids) == predictions
    assert _forest_state(restored) == _forest_state(classifier)


def _directional_distance(first, second):
    return np.abs(first - second).sum() + (first[0] > second[0]) * 0.75


@pytest.mark.parametrize("model_type", MODELS)
def test_asymmetric_indexed_distances_preserve_orientation_and_forest_state(model_type):
    features = np.asarray([[0.0], [10.0], [5.03], [0.07], [10.1], [0.1], [9.97]])
    train_ids = np.asarray([5, 1, 4, 0])
    query_ids = np.asarray([3, 6, 2])
    labels = np.asarray([0, 1, 1, 0])
    distances = np.asarray([[_directional_distance(first, second) for second in features] for first in features])
    direct = model_type(distance="euclidean")
    direct.distance_fn = _directional_distance
    precomputed = model_type(distance="euclidean")
    precomputed.pre_computed_distance = True
    precomputed.pre_distances = distances

    for classifier in (direct, precomputed):
        _fit(classifier, features[train_ids], labels, train_ids)

    assert _forest_state(precomputed) == _forest_state(direct)
    assert precomputed.predict(features[query_ids], query_ids) == direct.predict(features[query_ids], query_ids)
    assert _forest_state(precomputed) == _forest_state(direct)


@pytest.mark.parametrize("model_type", [KNNSupervisedOPF, UnsupervisedOPF])
@pytest.mark.parametrize("best_k", [1, 2, 3])
@pytest.mark.parametrize("precomputed", [False, True])
def test_knn_prediction_matches_stable_neighbour_reference(model_type, best_k, precomputed):
    features = np.asarray([[-2.0], [2.0], [-2.0], [6.0]])
    queries = np.asarray([[0.0], [5.0], [0.0], [-3.0]])
    classifier = model_type(distance="euclidean")
    classifier.subgraph = KNNSubgraph(features)
    classifier.subgraph.best_k = best_k
    classifier.subgraph.constant = 3.0
    classifier.subgraph.min_density = 0.01
    classifier.subgraph.max_density = 0.8
    classifier.subgraph.trained = True
    for index, node in enumerate(classifier.subgraph.nodes):
        node.cost = [10.0, 10.0, 50.0, 5.0][index]
        node.predicted_label = [2, 1, 3, 4][index]
        node.cluster_label = index
    if precomputed:
        classifier.pre_computed_distance = True
        classifier.pre_distances = np.abs(queries - features.T)

    expected_labels = []
    expected_clusters = []
    for query in queries:
        neighbours = sorted(range(len(features)), key=lambda index: np.abs(query - features[index]).sum())[:best_k]
        density = sum(np.exp(-np.abs(query - features[index]).sum() / 3.0) for index in neighbours) / best_k
        density = (constants.MAX_DENSITY - 1) * (density - 0.01) / (0.8 - 0.01 + constants.EPSILON) + 1
        winner = max(neighbours, key=lambda index: min(classifier.subgraph.nodes[index].cost, density))
        expected_labels.append(classifier.subgraph.nodes[winner].predicted_label)
        expected_clusters.append(classifier.subgraph.nodes[winner].cluster_label)

    state = _forest_state(classifier)
    predictions = classifier.predict(queries)

    assert predictions == ((expected_labels, expected_clusters) if model_type is UnsupervisedOPF else expected_labels)
    assert expected_labels[0] == expected_labels[2]
    assert _forest_state(classifier) == state


@pytest.mark.parametrize("model_type", MODELS)
@pytest.mark.parametrize("readonly", [False, True])
def test_fit_and_prediction_leave_borrowed_array_views_unchanged(model_type, readonly):
    storage = np.asarray([[0.0], [99.0], [0.1], [99.0], [10.0], [99.0], [10.1], [99.0]])
    features = storage[::2]
    labels = np.asarray([0, 0, 1, 1])
    indexes = np.arange(4)
    distances = np.abs(features - features.T)
    originals = [array.copy() for array in (storage, labels, indexes, distances)]
    for array in (features, labels, indexes, distances):
        array.flags.writeable = not readonly
    classifier = model_type(distance="euclidean")
    classifier.pre_computed_distance = True
    classifier.pre_distances = distances

    _fit(classifier, features, labels, indexes)
    classifier.predict(features, indexes)

    for array, original in zip((storage, labels, indexes, distances), originals):
        np.testing.assert_array_equal(array, original)
