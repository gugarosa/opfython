import json

import numpy as np
import pytest

from opfython.utils import converter


@pytest.mark.parametrize(
    ("function", "suffix"),
    [
        (converter.opf2txt, ".txt"),
        (converter.opf2csv, ".csv"),
        (converter.opf2json, ".json"),
    ],
)
def test_opf_converters(function, suffix, tmp_path):
    output = tmp_path / f"boat{suffix}"
    function("data/boat.dat", output)

    if suffix == ".json":
        with output.open(encoding="utf-8") as json_file:
            assert len(json.load(json_file)["data"]) == 100
    else:
        delimiter = "," if suffix == ".csv" else None
        assert np.loadtxt(output, delimiter=delimiter).shape == (100, 4)
