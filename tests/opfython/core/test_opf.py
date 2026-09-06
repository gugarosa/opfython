# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import pytest

from opfython.core import OPF, Subgraph
from opfython.math import distance
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


def test_opf_named_distance_updates_its_callable():
    classifier = OPF(distance="euclidean")
    classifier.subgraph = Subgraph(np.asarray([[0.0, 0.0], [3.0, 4.0]]))

    assert classifier.get_distances()[0, 1] == 5.0
    classifier.distance = "manhattan"

    assert classifier.distance == "manhattan"
    assert classifier.distance_fn is distance.manhattan_distance
    assert classifier.get_distances()[0, 1] == 7.0


@pytest.mark.parametrize("invalid", [None, [], "unknown"])
def test_opf_invalid_metric_update_preserves_configuration(invalid):
    classifier = OPF(distance="euclidean")
    original = classifier.distance_fn

    with pytest.raises(exception.TypeError):
        classifier.distance = invalid

    assert classifier.distance == "euclidean"
    assert classifier.distance_fn is original


def test_opf_invalid_registered_callable_preserves_configuration(monkeypatch):
    classifier = OPF(distance="euclidean")
    original = classifier.distance_fn
    monkeypatch.setitem(distance.DISTANCES, "invalid_callable", None)

    with pytest.raises(exception.TypeError):
        classifier.distance = "invalid_callable"

    assert classifier.distance == "euclidean"
    assert classifier.distance_fn is original


def test_opf_custom_callable_remains_active_until_a_named_metric_is_selected():
    classifier = OPF(distance="euclidean")
    classifier.subgraph = Subgraph(np.asarray([[0.0, 0.0], [3.0, 4.0]]))
    classifier.distance_fn = lambda left, right: 42.0

    assert classifier.get_distances()[0, 1] == 42.0

    classifier.distance = "manhattan"

    assert classifier.get_distances()[0, 1] == 7.0


def test_opf_get_distances_requires_a_subgraph():
    with pytest.raises(exception.BuildError, match=r"`subgraph` is None\."):
        OPF().get_distances()
