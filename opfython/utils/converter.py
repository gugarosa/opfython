# Copyright (c) 2020-2026 Gustavo de Rosa.
# Licensed under the Apache License, Version 2.0.

"""Convert OPF binary data to text formats."""

import json
import struct
from os import PathLike
from pathlib import Path

import numpy as np

from opfython.utils.logging import get_logger

logger = get_logger(__name__)


def _read_opf(opf_path: str | PathLike[str]) -> list[tuple[int | float, ...]]:
    with open(opf_path, "rb") as opf_file:
        header = struct.Struct("<iii")
        n_samples, _, n_features = header.unpack(opf_file.read(header.size))
        sample = struct.Struct(f"<ii{n_features}f")
        return [sample.unpack(opf_file.read(sample.size)) for _ in range(n_samples)]


def _output_path(
    opf_path: str | PathLike[str], output_file: str | PathLike[str] | None, suffix: str
) -> str | PathLike[str]:
    return output_file or str(Path(opf_path).with_suffix(suffix))


def opf2txt(opf_path: str | PathLike[str], output_file: str | PathLike[str] | None = None) -> None:
    """Convert an OPF binary file to whitespace-separated text with zero-based labels.

    Args:
        opf_path: Source binary file containing sample IDs, one-based labels, and features.
        output_file: Destination file, or None to replace the source suffix with .txt.

    Raises:
        OSError: The source cannot be read or the destination cannot be written.
        struct.error: A binary header or sample is incomplete.

    """

    logger.info("Converting file: %s ...", opf_path)
    samples = [(idx, label - 1, *features) for idx, label, *features in _read_opf(opf_path)]

    output_file = _output_path(opf_path, output_file, ".txt")
    np.savetxt(output_file, samples, delimiter=" ")
    logger.info("File converted to %s.", output_file)


def opf2csv(opf_path: str | PathLike[str], output_file: str | PathLike[str] | None = None) -> None:
    """Convert an OPF binary file to CSV with zero-based labels.

    Args:
        opf_path: Source binary file containing sample IDs, one-based labels, and features.
        output_file: Destination file, or None to replace the source suffix with .csv.

    Raises:
        OSError: The source cannot be read or the destination cannot be written.
        struct.error: A binary header or sample is incomplete.

    """

    logger.info("Converting file: %s ...", opf_path)
    samples = [(idx, label - 1, *features) for idx, label, *features in _read_opf(opf_path)]

    output_file = _output_path(opf_path, output_file, ".csv")
    np.savetxt(output_file, samples, delimiter=",")
    logger.info("File converted to %s.", output_file)


def opf2json(opf_path: str | PathLike[str], output_file: str | PathLike[str] | None = None) -> None:
    """Convert an OPF binary file to JSON with zero-based labels under a data key.

    Args:
        opf_path: Source binary file containing sample IDs, one-based labels, and features.
        output_file: Destination file, or None to replace the source suffix with .json.

    Raises:
        OSError: The source cannot be read or the destination cannot be written.
        struct.error: A binary header or sample is incomplete.

    """

    logger.info("Converting file: %s ...", opf_path)
    records = [{"id": idx, "label": label - 1, "features": features} for idx, label, *features in _read_opf(opf_path)]

    output_file = _output_path(opf_path, output_file, ".json")
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump({"data": records}, json_file)

    logger.info("File converted to %s.", output_file)
