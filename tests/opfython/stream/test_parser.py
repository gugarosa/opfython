# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

from unittest.mock import Mock

import numpy as np
import pytest

from opfython.stream import loader, parser
from opfython.utils import exception


def test_parse_loader():
    X, Y = parser.parse_loader(loader.load_csv("data/boat.csv"))

    assert X.shape == (100, 2)
    assert Y.shape == (100,)


def test_parse_loader_handles_non_array_input():
    assert parser.parse_loader([]) == (None, None)
    assert parser.parse_loader(None) == (None, None)


def test_parse_loader_rejects_nonsequential_labels():
    data = np.asarray([[0, 0, 1], [1, 2, 1]])

    with pytest.raises(exception.ValueError):
        parser.parse_loader(data)


def test_parse_loader_preserves_feature_views_and_copies_labels():
    data = np.array([[7, 0, 1.5], [8, 1, 2.5]])
    original = data.copy()

    features, labels = parser.parse_loader(data)

    np.testing.assert_array_equal(data, original)
    np.testing.assert_array_equal(features, [[1.5], [2.5]])
    np.testing.assert_array_equal(labels, [0, 1])
    assert np.shares_memory(features, data)
    assert not np.shares_memory(labels, data)
    assert np.issubdtype(labels.dtype, np.integer)


def test_parse_loader_warns_for_a_single_label(monkeypatch):
    warning = Mock()
    monkeypatch.setattr(parser.logger, "warning", warning)

    _, labels = parser.parse_loader(np.array([[0, 0, 1], [1, 0, 2]]))

    np.testing.assert_array_equal(labels, [0, 0])
    format_string, *arguments = warning.call_args.args
    diagnostic = format_string % tuple(arguments)
    assert "`n_labels=1`" in diagnostic
    assert diagnostic.endswith(".")
