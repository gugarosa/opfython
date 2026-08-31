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
