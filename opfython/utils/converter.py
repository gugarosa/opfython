"""Convert OPF binary data to text formats."""

import json
import struct
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from opfython.utils import logging

logger = logging.get_logger(__name__)


def _read_opf(opf_path: str) -> List[Tuple]:
    with open(opf_path, "rb") as opf_file:
        header = struct.Struct("<iii")
        n_samples, _, n_features = header.unpack(opf_file.read(header.size))
        sample = struct.Struct(f"<ii{n_features}f")
        return [sample.unpack(opf_file.read(sample.size)) for _ in range(n_samples)]


def _output_path(opf_path: str, output_file: Optional[str], suffix: str) -> str:
    return output_file or str(Path(opf_path).with_suffix(suffix))


def opf2txt(opf_path: str, output_file: Optional[str] = None) -> None:
    """Convert an OPF binary file to whitespace-separated text."""

    logger.info("Converting file: %s ...", opf_path)
    samples = [
        (idx, label - 1, *features) for idx, label, *features in _read_opf(opf_path)
    ]
    output_file = _output_path(opf_path, output_file, ".txt")
    np.savetxt(output_file, samples, delimiter=" ")
    logger.info("File converted to %s.", output_file)


def opf2csv(opf_path: str, output_file: Optional[str] = None) -> None:
    """Convert an OPF binary file to CSV."""

    logger.info("Converting file: %s ...", opf_path)
    samples = [
        (idx, label - 1, *features) for idx, label, *features in _read_opf(opf_path)
    ]
    output_file = _output_path(opf_path, output_file, ".csv")
    np.savetxt(output_file, samples, delimiter=",")
    logger.info("File converted to %s.", output_file)


def opf2json(opf_path: str, output_file: Optional[str] = None) -> None:
    """Convert an OPF binary file to JSON."""

    logger.info("Converting file: %s ...", opf_path)
    records = [
        {"id": idx, "label": label - 1, "features": features}
        for idx, label, *features in _read_opf(opf_path)
    ]
    output_file = _output_path(opf_path, output_file, ".json")
    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as json_file:
        json.dump({"data": records}, json_file)
    logger.info("File converted to %s.", output_file)
