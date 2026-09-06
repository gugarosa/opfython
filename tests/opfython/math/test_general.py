# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

import numpy as np
import pytest

from opfython.math import general
from opfython.stream import loader, parser, splitter
from opfython.utils import exception


def test_confusion_matrix():
    labels = [0, 0, 1, 1]
    preds = [0, 0, 1, 1]

    c_matrix = general.confusion_matrix(labels, preds)

    assert c_matrix.shape == (2, 2)


def test_normalize():
    array = [1, 1, 1, 2]

    norm_array = general.normalize(array)

    assert norm_array[3] == 1.7320508075688774


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("storage", ["writable", "readonly", "view"])
def test_normalize_preserves_input_storage_and_dtype(dtype, storage):
    base = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=dtype)
    original = base.copy()
    array = base[::2] if storage == "view" else base
    if storage == "readonly":
        array.setflags(write=False)

    normalized = general.normalize(array)

    assert normalized.dtype == dtype
    assert normalized.shape == array.shape
    np.testing.assert_array_equal(base, original)
    np.testing.assert_allclose(normalized.mean(axis=0), 0, atol=1e-7)
    np.testing.assert_allclose(normalized.std(axis=0), 1, atol=1e-7)


def test_normalize_preserves_constant_input_and_rejects_scalars():
    with pytest.raises(IndexError, match="axis 0"):
        general.normalize(1.0)

    with np.errstate(invalid="ignore"):
        constant = general.normalize((1.0, 1.0))

    np.testing.assert_array_equal(constant, [np.nan, np.nan])


def test_opf_accuracy():
    labels = [0, 0, 1, 1]
    preds = [0, 0, 0, 0]

    acc = general.opf_accuracy(labels, preds)

    assert acc == 0.5


def test_opf_accuracy_per_label():
    labels = [0, 0, 1, 1]
    preds = [0, 0, 0, 0]

    acc_per_label = general.opf_accuracy_per_label(labels, preds)

    assert acc_per_label.shape == (2,)


def test_opf_pre_compute_distances(tmp_path):
    txt = loader.load_txt("data/boat.txt")

    X, Y = parser.parse_loader(txt)

    X_train, _, _, _ = splitter.split(X, Y, 0.5, 1)

    output = tmp_path / "distances.txt"
    general.pre_compute_distance(X_train, output, "log_squared_euclidean")

    assert output.is_file()


def test_purity():
    labels = [0, 0, 1, 1]
    preds = [0, 0, 1, 1]

    purity = general.purity(labels, preds)

    assert purity == 1


@pytest.mark.parametrize(
    ("clusters", "expected"),
    [
        ([0, 1, 2, 3], 1.0),
        ([40, 40, 10, 10], 1.0),
        ([7, 8, 7, 8], 0.5),
        ([0, 0, 0, 0], 0.5),
    ],
)
def test_purity_is_independent_of_cluster_numbering(clusters, expected):
    assert general.purity([0, 0, 1, 1], clusters) == expected


@pytest.mark.parametrize(
    "metric",
    [
        general.confusion_matrix,
        general.opf_accuracy,
        general.opf_accuracy_per_label,
        general.purity,
    ],
)
@pytest.mark.parametrize("predictions", [[0, 1], [0, 1, 0, 1], [[0, 1, 0]], 0])
def test_metrics_reject_invalid_label_shapes(metric, predictions):
    with pytest.raises(exception.SizeError, match=r"`labels` and `preds` .* samples\.$"):
        metric([0, 1, 0], predictions)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (general.confusion_matrix, [[1, 0], [1, 0]]),
        (general.opf_accuracy, 0.5),
        (general.opf_accuracy_per_label, [1.0, 0.0]),
        (general.purity, 0.5),
    ],
)
def test_metrics_accept_tuple_labels(metric, expected):
    actual = metric((0, 1), (0, 0))

    np.testing.assert_array_equal(actual, expected)


def test_opf_accuracy_handles_a_single_class_without_invalid_division():
    with np.errstate(divide="raise", invalid="raise"):
        assert general.opf_accuracy([0, 0], [0, 0]) == 1.0


def test_confusion_matrix_includes_predicted_classes_absent_from_labels():
    np.testing.assert_array_equal(
        general.confusion_matrix([0, 0], [0, 1]),
        [[1, 1], [0, 0]],
    )


@pytest.mark.parametrize(("predictions", "expected"), [([1, 1], 0.5), ([0, 1], 0.75)])
def test_opf_accuracy_is_invariant_to_class_numbering(predictions, expected):
    with np.errstate(divide="raise", invalid="raise"):
        original = general.opf_accuracy([0, 0], predictions)
        relabeled = general.opf_accuracy([1, 1], [1 - p for p in predictions])

    assert original == relabeled == expected


@pytest.mark.parametrize(
    ("labels", "predictions"),
    [([0, 2, 2], [0, 0, 2]), ([2, 2], [0, 2])],
)
def test_opf_accuracy_per_label_preserves_class_indexes(labels, predictions):
    with np.errstate(divide="raise", invalid="raise"):
        actual = general.opf_accuracy_per_label(labels, predictions)

    np.testing.assert_array_equal(actual, [1.0, 1.0, 0.5])
