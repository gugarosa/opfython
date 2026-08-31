# OPFython

[![PyPI](https://img.shields.io/pypi/v/opfython.svg)](https://pypi.org/project/opfython/)
[![CI](https://github.com/gugarosa/opfython/actions/workflows/ci.yml/badge.svg)](https://github.com/gugarosa/opfython/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/opfython/badge/?version=latest)](https://opfython.readthedocs.io/)
[![DOI](https://img.shields.io/badge/DOI-10.1016/j.simpa.2021.100113-006DB9.svg)](https://doi.org/10.1016/j.simpa.2021.100113)
[![License](https://img.shields.io/github/license/gugarosa/opfython.svg)](LICENSE)

OPFython is a Python implementation of the Optimum-Path Forest family of
classifiers. It provides supervised, semi-supervised, unsupervised, and
KNN-supervised models backed by NumPy and Numba.

This implementation follows [LibOPF](https://github.com/jppbsi/LibOPF).
Please cite the original LibOPF authors as well as OPFython when using it in
research.

OPFython requires Python 3.11 or newer.

## Installation

```bash
uv add opfython
```

## Quick start

```python
import numpy as np

from opfython.models import SupervisedOPF

X_train = np.asarray([[0.0, 0.0], [0.1, 0.2], [1.0, 1.0], [1.1, 0.9]])
Y_train = np.asarray([0, 0, 1, 1])
X_test = np.asarray([[0.05, 0.1], [1.05, 1.0]])

classifier = SupervisedOPF()
classifier.fit(X_train, Y_train)
predictions = classifier.predict(X_test)
```

Labels must be zero-based and sequential. Pre-computed distance matrices can
be supplied through each classifier's `pre_computed_distance` constructor
argument.

## Classifiers

| Class | Purpose |
|---|---|
| `SupervisedOPF` | Complete-graph supervised classification |
| `KNNSupervisedOPF` | Supervised classification with learned KNN adjacency |
| `SemiSupervisedOPF` | Learning from labeled and unlabeled samples |
| `UnsupervisedOPF` | Density-based clustering and label propagation |

The package also includes 47 distance metrics, random generators, OPF
evaluation measures, dataset loaders and splitters, package logging and
exception helpers, and converters for LibOPF binary datasets.

See [the documentation](https://opfython.readthedocs.io/) and the
[`examples/applications`](examples/applications) directory for complete
workflows.

## Development

```bash
uv sync --all-groups
uv run pytest
uv run pre-commit run --all-files
uv run --group docs sphinx-build -W -b html docs docs/_build/html
uv build --no-sources
```

## Citation

```bibtex
@article{rosa2021simpa,
    title = {OPFython: A Python implementation for Optimum-Path Forest},
    author = {Gustavo H. {de Rosa} and Joao P. Papa},
    journal = {Software Impacts},
    pages = {100113},
    year = {2021},
    issn = {2665-9638},
    doi = {https://doi.org/10.1016/j.simpa.2021.100113}
}
```

```bibtex
@misc{rosa2021speedup,
    title = {Speeding Up OPFython with Numba},
    author = {Gustavo H. de Rosa and Joao Paulo Papa},
    year = {2021},
    eprint = {2106.11828},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

OPFython is licensed under the Apache License 2.0.
