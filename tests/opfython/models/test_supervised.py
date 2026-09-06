import numpy as np
import pytest

from opfython.models import SupervisedOPF
from opfython.stream import loader, parser, splitter
from opfython.utils import constants, exception

X, Y = parser.parse_loader(loader.load_csv("data/boat.csv"))


def test_supervised_requires_fitted_subgraph():
    classifier = SupervisedOPF()

    with pytest.raises(exception.BuildError):
        classifier.predict(X)


def test_supervised_fit_and_predict():
    classifier = SupervisedOPF()

    classifier.fit(X, Y)
    predictions = classifier.predict(X)

    assert classifier.subgraph.trained is True
    assert len(predictions) == 100


def test_supervised_uses_precomputed_distances():
    classifier = SupervisedOPF()
    classifier.pre_computed_distance = True
    classifier.pre_distances = np.ones((100, 100))

    classifier.fit(X, Y)

    assert len(classifier.predict(X)) == 100


def test_supervised_learn():
    classifier = SupervisedOPF()
    X_train, X_val, Y_train, Y_val = splitter.split(
        X,
        Y,
        percentage=0.1,
        random_state=1,
    )

    classifier.learn(X_train, Y_train, X_val, Y_val, n_iterations=5)

    assert classifier.subgraph.trained is True


def test_supervised_prune():
    classifier = SupervisedOPF()
    X_train, X_val, Y_train, Y_val = splitter.split(
        X,
        Y,
        percentage=0.1,
        random_state=1,
    )

    classifier.prune(X_train, Y_train, X_val, Y_val, n_iterations=5)

    assert classifier.subgraph.n_nodes == 10


def test_supervised_prune_retains_every_winning_prototype():
    features = np.asarray([[0.0], [10.0]])
    labels = np.asarray([0, 1])
    classifier = SupervisedOPF(distance="euclidean")

    classifier.fit(features, labels)
    assert classifier.predict(features) == labels.tolist()
    assert all(
        node.relevant == constants.RELEVANT for node in classifier.subgraph.nodes
    )

    classifier.prune(features, labels, features, labels, n_iterations=2)

    assert classifier.subgraph.n_nodes == 2
    assert classifier.predict(features) == labels.tolist()


@pytest.mark.parametrize("n_samples", [1, 3])
def test_supervised_fits_single_class(n_samples):
    features = np.arange(n_samples, dtype=float).reshape(-1, 1)
    labels = np.zeros(n_samples, dtype=int)
    classifier = SupervisedOPF(distance="euclidean")

    classifier.fit(features, labels)

    assert classifier.predict(np.asarray([[-1.0], [4.0]])) == [0, 0]
    assert sorted(classifier.subgraph.idx_nodes) == list(range(n_samples))


def test_supervised_pruning_can_retain_a_single_class():
    features = np.asarray([[0.0], [10.0]])
    labels = np.asarray([0, 1])
    classifier = SupervisedOPF(distance="euclidean")

    with np.errstate(divide="raise", invalid="raise"):
        classifier.prune(features, labels, features[:1], labels[:1], n_iterations=2)

    assert classifier.subgraph.n_nodes == 1
    assert classifier.predict(features[:1]) == [0]
