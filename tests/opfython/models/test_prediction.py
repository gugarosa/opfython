import numpy as np
import pytest

from opfython.models import (
    KNNSupervisedOPF,
    SemiSupervisedOPF,
    SupervisedOPF,
    UnsupervisedOPF,
)
from opfython.utils import exception

MODELS = [SupervisedOPF, SemiSupervisedOPF, KNNSupervisedOPF, UnsupervisedOPF]


def _fit(classifier, features, labels, indexes=None):
    if isinstance(classifier, KNNSupervisedOPF):
        classifier.fit(
            features, labels, features, labels, I_train=indexes, I_val=indexes
        )
    elif isinstance(classifier, SemiSupervisedOPF):
        classifier.fit(
            features, labels, np.empty((0, features.shape[1])), I_train=indexes
        )
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
def test_knn_predictions_do_not_depend_on_query_batch(
    model_type, precomputed, queries, expected, tmp_path
):
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
    singletons = [
        classifier.predict(features[index : index + 1], np.asarray([index]))
        for index in query_ids
    ]
    reversed_batch = classifier.predict(features[query_ids[::-1]], query_ids[::-1])

    if model_type is UnsupervisedOPF:
        assert batch[0] == [result[0][0] for result in singletons] == expected
        assert batch[1] == [result[1][0] for result in singletons]
        assert reversed_batch == (batch[0][::-1], batch[1][::-1])
    else:
        assert batch == [result[0] for result in singletons] == expected
        assert reversed_batch == batch[::-1]


@pytest.mark.parametrize("model_type", MODELS)
def test_precomputed_distances_match_direct_predictions_for_sample_ids(
    model_type, tmp_path
):
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

    assert precomputed.predict(features[test_ids], test_ids) == direct.predict(
        features[test_ids], test_ids
    )


@pytest.mark.parametrize("model_type", MODELS)
@pytest.mark.parametrize("shape", [(3, 3), (6, 3), (3,), ()])
def test_models_reject_distance_matrices_that_do_not_cover_training_ids(
    model_type, shape
):
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

    assert precomputed.predict(features[4:], np.asarray([4, 5])) == direct.predict(
        features[4:]
    )


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
