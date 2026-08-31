import numpy as np
import pytest

from opfython.models import SupervisedOPF
from opfython.stream import loader, parser, splitter
from opfython.utils import exception

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
