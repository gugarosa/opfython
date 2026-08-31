import numpy as np
import pytest

from opfython.models import UnsupervisedOPF
from opfython.stream import loader, parser
from opfython.utils import exception

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
