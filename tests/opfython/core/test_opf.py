import numpy as np
import pytest

from opfython.core import OPF, Subgraph
from opfython.utils import exception


def test_opf_defaults():
    classifier = OPF()

    assert classifier.subgraph is None
    assert classifier.distance == "log_squared_euclidean"
    assert callable(classifier.distance_fn)
    assert classifier.pre_computed_distance is False
    assert classifier.pre_distances is None


def test_opf_rejects_unknown_distance():
    with pytest.raises(exception.TypeError):
        OPF(distance="unknown")


def test_opf_reads_precomputed_distances():
    classifier = OPF(pre_computed_distance="data/boat.txt")

    assert classifier.pre_computed_distance is True
    assert classifier.pre_distances.shape == (100, 4)

    with pytest.raises(exception.ArgumentError):
        classifier._read_distances("data/boat.json")


def test_opf_gets_distances():
    classifier = OPF(distance="euclidean")
    classifier.subgraph = Subgraph(
        np.asarray([[0.0, 0.0], [3.0, 4.0]]),
        np.asarray([0, 1]),
    )

    assert np.array_equal(
        classifier.get_distances(),
        np.asarray([[0.0, 5.0], [5.0, 0.0]]),
    )


def test_opf_round_trip(tmp_path):
    path = tmp_path / "model.pkl"
    classifier = OPF(distance="bray_curtis")

    classifier.save(path)
    loaded = OPF()
    loaded.load(path)

    assert loaded.distance == "bray_curtis"


def test_opf_requires_concrete_fit_and_predict():
    classifier = OPF()

    with pytest.raises(NotImplementedError):
        classifier.fit(None, None)
    with pytest.raises(NotImplementedError):
        classifier.predict(None)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("subgraph", "invalid"),
        ("distance", "invalid"),
        ("distance_fn", "invalid"),
        ("pre_computed_distance", "invalid"),
        ("pre_distances", "invalid"),
    ],
)
def test_opf_validates_public_attributes(attribute, value):
    classifier = OPF()

    with pytest.raises(exception.TypeError):
        setattr(classifier, attribute, value)


def test_opf_rejects_unhashable_distance():
    with pytest.raises(exception.TypeError):
        OPF(distance=[])
