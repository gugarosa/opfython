# OPFython architecture

OPFython is a small NumPy/Numba package organized around four classifier
implementations and the graph structures they share.

## Package map

```text
opfython/
|-- core/
|   |-- heap.py       Mutable min/max priority queue
|   |-- node.py       Mutable sample state
|   |-- opf.py        Shared classifier behavior
|   `-- subgraph.py   Node collection and dataset construction
|-- math/
|   |-- distance.py   Numba-compiled distance metrics
|   |-- general.py    Accuracy, normalization, and purity helpers
|   `-- random.py     Uniform and Gaussian generators
|-- models/
|   |-- _common.py   Shared private prediction and forest-growth operations
|   |-- supervised.py
|   |-- knn_supervised.py
|   |-- semi_supervised.py
|   `-- unsupervised.py
|-- stream/
|   |-- loader.py     CSV, text, and JSON readers
|   |-- parser.py     OPF row parsing
|   `-- splitter.py   Dataset split and merge helpers
|-- subgraphs/
|   `-- knn.py        K-nearest-neighbour graph construction
`-- utils/
    |-- constants.py
    |-- converter.py  LibOPF binary conversion
    |-- decorator.py  Numerical guard used by distance functions
    |-- exception.py  Package exception hierarchy
    `-- logging.py    Console and rotating-file logger helpers
```

## Data flow

1. `stream.loader` reads rows whose first columns are sample id and label.
2. `stream.parser.parse_loader` returns feature and label arrays.
3. A model builds a `Subgraph` or `KNNSubgraph` containing `Node` instances.
4. OPF competition uses `Heap` for mutable min/max priorities.
5. Fitted models expose predictions, cluster assignments, or propagated
   labels.

Labels are zero-based and sequential. Optional sample indexes are retained so
models can address pre-computed distance matrices. Feature, label, and index
counts are checked before building nodes.

## Core structures

`Node` stores the mutable state used by the algorithms: labels, features,
cost, density, adjacency, predecessor, root, and relevance flags.
Its public properties retain the package's mutation-time type and value
validation.

`Subgraph.n_nodes` is derived from `len(nodes)` so graph size cannot drift from
the actual population. `KNNSubgraph` adds density bounds, cluster count, and
the selected neighbourhood size. Rebuilding arcs replaces adjacency and
resets the distance scale. Each clustering pass replaces its node visit order.

Unsupervised neighbourhood selection computes the maximum-k arcs once.
Plateau-only reverse edges are removed between candidate evaluations so the
next candidate uses the original nearest-neighbour ordering.

`Heap` keeps mutable priorities and stable tie behavior required by the OPF
algorithms while supporting both minimum and maximum policies.

`OPF` resolves a distance name through `math.distance.DISTANCES`, optionally
loads a pre-computed matrix, and provides serialization plus pairwise distance
generation. Indexed matrix access is validated on the row and column axes
used by each model. Concrete models implement `fit` and `predict`. Package
logging and custom exception types remain available for applications that
rely on them.

The KNN models share a private prediction kernel while retaining their distinct
public returns. Supervised and semi-supervised models share minimax forest growth,
with semi-supervised label propagation kept explicit. Private operations do not
change model class identities or persisted graph fields.

## Classifiers

- `SupervisedOPF` finds prototypes with a minimum spanning tree and propagates
  labels through minimum-cost paths.
- `KNNSupervisedOPF` selects the best neighbourhood size on validation data
  before density-based classification.
- `SemiSupervisedOPF` extends supervised training with unlabeled nodes.
- `UnsupervisedOPF` selects a neighbourhood through normalized cut and forms
  density-based clusters.

## Tooling

Project metadata, dependency groups, test configuration, and build settings
live in `pyproject.toml`. [CONVENTIONS.md](CONVENTIONS.md) defines the project
style and compatibility rules. GitHub Actions tests Python 3.11 through 3.13,
runs the existing quality hooks, and builds the documentation and its examples.
Releases publish built artifacts to PyPI.
