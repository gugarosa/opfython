# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import json
import sys

import numpy as np
import pytest

from opfython.stream import loader


@pytest.mark.parametrize(
    ("function", "path"),
    [
        (loader.load_csv, "data/boat.csv"),
        (loader.load_json, "data/boat.json"),
        (loader.load_txt, "data/boat.txt"),
    ],
)
def test_loaders(function, path):
    assert function(path).shape == (100, 4)


@pytest.mark.parametrize(
    "function",
    [loader.load_csv, loader.load_json, loader.load_txt],
)
def test_loaders_return_none_for_missing_files(function):
    assert function("data/missing") is None


@pytest.mark.parametrize("contents", [b"{", b"\xff", b"{}", b"[]", b"null", b"42", b'"text"'])
def test_load_json_returns_none_for_decode_and_top_level_schema_failures(contents, tmp_path):
    path = tmp_path / "invalid.json"
    path.write_bytes(contents)

    assert loader.load_json(path) is None


@pytest.mark.parametrize(
    ("records", "error"),
    [
        ([{"id": 0, "features": [1]}], KeyError),
        ([{"id": 0, "label": 0, "features": None}], TypeError),
        ([None], TypeError),
        (None, TypeError),
        ([{"id": 0, "label": 0, "features": [1]}, {"id": 1, "label": 0, "features": [1, 2]}], ValueError),
    ],
)
def test_load_json_propagates_malformed_record_failures(records, error, tmp_path):
    path = tmp_path / "records.json"
    path.write_text(json.dumps({"data": records}), encoding="utf-8")

    with pytest.raises(error):
        loader.load_json(path)


def test_load_json_propagates_unexpected_errors(monkeypatch, tmp_path):
    path = tmp_path / "valid.json"
    path.write_text('{"data": []}', encoding="utf-8")

    def fail_decode(stream):
        raise RuntimeError("Unexpected decoder failure")

    monkeypatch.setattr(loader.json, "load", fail_decode)

    with pytest.raises(RuntimeError, match="Unexpected decoder failure"):
        loader.load_json(path)


def test_load_json_returns_none_for_integer_decoding_failure(tmp_path):
    path = tmp_path / "integer.json"
    path.write_text('{"data": ' + "1" * 641 + "}", encoding="utf-8")
    previous_limit = sys.get_int_max_str_digits()

    sys.set_int_max_str_digits(640)
    try:
        assert loader.load_json(path) is None
    finally:
        sys.set_int_max_str_digits(previous_limit)


def test_load_json_preserves_inferred_data_and_empty_shape(tmp_path):
    path = tmp_path / "records.json"
    path.write_text('{"data": [{"id": 7, "label": 2, "features": [1.5, 3]}]}', encoding="utf-8")

    np.testing.assert_array_equal(loader.load_json(path), [[7, 2, 1.5, 3]])

    path.write_text('{"data": []}', encoding="utf-8")

    assert loader.load_json(path).shape == (0,)


@pytest.mark.parametrize("function", [loader.load_csv, loader.load_txt])
def test_numeric_loaders_propagate_malformed_content(function, tmp_path):
    path = tmp_path / "invalid.txt"
    path.write_text("not numeric", encoding="utf-8")

    with pytest.raises(ValueError):
        function(path)
