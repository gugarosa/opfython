# OPFython — Architecture Guide

> **Version:** 1.0.14 | **License:** Apache 2.0 | **Python:** 3.6+
> **DOI:** [10.1016/j.simpa.2021.100113](https://doi.org/10.1016/j.simpa.2021.100113)

OPFython is a pure-Python implementation of the **Optimum-Path Forest (OPF)** family of classifiers, originally developed in LibOPF (C library). It provides supervised, semi-supervised, unsupervised, and KNN-supervised variants of OPF, backed by Numba JIT-compiled distance functions and NumPy arrays. The package follows a layered architecture where data flows from file I/O through graph construction into classifier algorithms.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Package Structure](#2-package-structure)
3. [Module Deep-Dive](#3-module-deep-dive)
   - 3.1 [core — Foundational Data Structures](#31-core--foundational-data-structures)
   - 3.2 [math — Mathematical Primitives](#32-math--mathematical-primitives)
   - 3.3 [models — Classifier Implementations](#33-models--classifier-implementations)
   - 3.4 [stream — Data I/O Pipeline](#34-stream--data-io-pipeline)
   - 3.5 [subgraphs — Specialized Graph Structures](#35-subgraphs--specialized-graph-structures)
   - 3.6 [utils — Cross-Cutting Utilities](#36-utils--cross-cutting-utilities)
4. [Class Hierarchy](#4-class-hierarchy)
5. [Data Flow](#5-data-flow)
6. [Distance Metrics Catalog](#6-distance-metrics-catalog)
7. [Key Algorithms](#7-key-algorithms)
8. [Serialization & Persistence](#8-serialization--persistence)
9. [Dependencies](#9-dependencies)

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          User Code / Examples                       │
├─────────────────────────────────────────────────────────────────────┤
│                        models (Classifiers)                         │
│  SupervisedOPF │ KNNSupervisedOPF │ SemiSupervisedOPF │ UnsupervisedOPF │
├─────────────────────────────────────────────────────────────────────┤
│       core (OPF, Subgraph, Node, Heap)     │   subgraphs (KNN)     │
├─────────────────────────────────────────────────────────────────────┤
│  stream (loader, parser, splitter)  │   math (distance, general)   │
├─────────────────────────────────────────────────────────────────────┤
│          utils (constants, converter, decorator, exception, logging)│
└─────────────────────────────────────────────────────────────────────┘
```

- **Bottom layer (utils):** Constants, custom exceptions, logging, decorators, and binary file converters shared across the entire package.
- **Data layer (stream + math):** File loading, parsing into NumPy arrays, train/test splitting, distance functions, normalization, and evaluation metrics.
- **Core layer (core + subgraphs):** The graph data structures — `Node`, `Subgraph`, `KNNSubgraph`, and `Heap` — that represent OPF's internal state.
- **Model layer (models):** Four concrete classifier classes that implement the OPF algorithm variants.
- **User layer:** Application scripts that compose the above.

---

## 2. Package Structure

```
opfython/                     # Root package (v1.0.14)
├── __init__.py
├── core/                     # Foundational data structures
│   ├── heap.py               #   Min/max priority queue
│   ├── node.py               #   Graph node with features, labels, costs
│   ├── opf.py                #   Abstract base classifier (OPF)
│   └── subgraph.py           #   Collection of Nodes (graph container)
├── math/                     # Mathematical building blocks
│   ├── distance.py           #   44 Numba-compiled distance metrics
│   ├── general.py            #   Accuracy, confusion matrix, normalization
│   └── random.py             #   Uniform & Gaussian random generators
├── models/                   # Classifier implementations
│   ├── supervised.py         #   SupervisedOPF (MST-based prototypes)
│   ├── knn_supervised.py     #   KNNSupervisedOPF (density-based with KNN)
│   ├── semi_supervised.py    #   SemiSupervisedOPF (labeled + unlabeled)
│   └── unsupervised.py       #   UnsupervisedOPF (clustering via KNN)
├── stream/                   # Data I/O pipeline
│   ├── loader.py             #   CSV, TXT, JSON file loading
│   ├── parser.py             #   OPF-format parsing (features + labels)
│   └── splitter.py           #   Train/test splitting & merging
├── subgraphs/                # Specialized subgraph variants
│   └── knn.py                #   KNNSubgraph with adjacency, PDF, arcs
└── utils/                    # Cross-cutting concerns
    ├── constants.py           #   EPSILON, FLOAT_MAX, colors, statuses
    ├── converter.py           #   Binary OPF → TXT/CSV/JSON converters
    ├── decorator.py           #   @avoid_zero_division decorator
    ├── exception.py           #   Custom error classes
    └── logging.py             #   Logger factory (console + file handler)
```

---

## 3. Module Deep-Dive

### 3.1 `core` — Foundational Data Structures

#### `Node` (`core/node.py`)
The atomic unit of OPF. Each node represents a single data sample in the graph.

| Attribute          | Type          | Description                                                |
|--------------------|---------------|------------------------------------------------------------|
| `idx`              | `int`         | Unique identifier                                           |
| `label`            | `int`         | Ground-truth class label                                    |
| `predicted_label`  | `int`         | Label assigned by the classifier                            |
| `cluster_label`    | `int`         | Cluster assignment (unsupervised mode)                      |
| `features`         | `np.ndarray`  | N-dimensional feature vector                                |
| `cost`             | `float`       | Path cost in the optimum-path forest                        |
| `density`          | `float`       | Probability density value (KNN-based models)                |
| `radius`           | `float`       | Maximum distance among k-nearest neighbours                 |
| `n_plateaus`       | `int`         | Count of adjacent nodes on density plateaus                 |
| `adjacency`        | `List[int]`   | List of adjacent node indices (KNN arcs)                    |
| `root`             | `int`         | Root node of this node's tree                               |
| `status`           | `int`         | `STANDARD` (0) or `PROTOTYPE` (1)                           |
| `pred`             | `int`         | Predecessor node index (`NIL` = -1 if root)                 |
| `relevant`         | `int`         | `IRRELEVANT` (0) or `RELEVANT` (1) — used in pruning        |

All attributes are guarded by property setters with type and value validation.

#### `Heap` (`core/heap.py`)
A binary heap (priority queue) supporting both **min-heap** and **max-heap** policies. It is the engine that drives Prim/Dijkstra-style algorithms in OPF.

| Component     | Description                                               |
|---------------|-----------------------------------------------------------|
| `cost[]`      | Priority values per node (initialized to `FLOAT_MAX`)      |
| `color[]`     | Node state: `WHITE` (unseen), `GRAY` (in heap), `BLACK` (removed) |
| `p[]`         | Mapping from heap position to node index                   |
| `pos[]`       | Mapping from node index to heap position                   |
| `policy`      | `"min"` for supervised OPF, `"max"` for density-based OPF  |

Key operations: `insert(p)`, `remove() → p`, `update(p, cost)`, `go_up(i)`, `go_down(i)`.

#### `Subgraph` (`core/subgraph.py`)
A container of `Node` objects representing a graph. Serves as the primary data holder for all OPF classifiers.

| Method/Attribute | Description                                                       |
|------------------|-------------------------------------------------------------------|
| `n_nodes`        | Computed from `len(self.nodes)` (dynamic property)                 |
| `n_features`     | Dimensionality of the feature space                                |
| `nodes`          | `List[Node]` — the graph vertices                                  |
| `idx_nodes`      | Ordered list of node indices (by cost, filled during training)      |
| `trained`        | Boolean flag indicating training completion                         |
| `_load(path)`    | Loads data from `.csv`, `.txt`, or `.json` files                    |
| `_build(X,Y,I)`  | Constructs `Node` objects from feature/label/index arrays           |
| `destroy_arcs()` | Removes all adjacency relationships                                 |
| `mark_nodes(i)`  | Marks a node and its predecessor chain as relevant (for pruning)    |
| `reset()`        | Resets predecessors, relevance flags, and arcs                      |

#### `OPF` (`core/opf.py`)
Abstract base class for all OPF classifiers. Handles:
- **Distance selection:** Validates and stores one of 44 distance metric names, resolves it to a callable from `math.distance.DISTANCES`.
- **Pre-computed distances:** Optionally loads a pre-computed distance matrix from file.
- **Serialization:** `save(path)` / `load(path)` via Python's `pickle`.
- **Distance matrix generation:** `get_distances(normalize=False)` computes a full N×N distance matrix.
- **Interface contract:** Declares `fit(X, Y)` and `predict(X)` as `NotImplementedError` — subclasses must override.

---

### 3.2 `math` — Mathematical Primitives

#### `distance.py` — 44 Distance Metrics
All distance functions accept two `np.array` vectors and return a `float`. Most are compiled with **Numba `@njit(cache=True)`** for near-C performance. Functions that risk division by zero are additionally wrapped with `@avoid_zero_division`, which adds `EPSILON` (1e-20) to both input arrays.

The `DISTANCES` dictionary at module bottom maps string keys to callables, enabling runtime distance selection.

**Categories of included distances:**

| Category                  | Metrics                                                                                      |
|---------------------------|----------------------------------------------------------------------------------------------|
| Lp Norms                  | `euclidean`, `squared_euclidean`, `manhattan`, `chebyshev`, `average_euclidean`               |
| Logarithmic Transforms    | `log_euclidean`, `log_squared_euclidean` (default), `lorentzian`                             |
| Statistical Divergences   | `kullback_leibler`, `jeffreys`, `jensen`, `jensen_shannon`, `k_divergence`, `topsoe`         |
| Chi-Squared Family        | `chi_squared`, `neyman`, `pearson`, `squared`, `additive_symmetric`, `divergence`            |
| Set/Similarity-Based      | `jaccard`, `dice`, `cosine`, `chord`, `hamming`                                               |
| Ecological Distances      | `bray_curtis`, `canberra`, `soergel`, `kulczynski`, `gower`                                  |
| Probabilistic             | `bhattacharyya`, `hellinger`, `matusita`, `squared_chord`, `hassanat`                        |
| Symmetric/Asymmetric      | `max_symmetric`, `min_symmetric`, `vicis_symmetric1/2/3`, `vicis_wave_hedges`, `sangvi`      |
| Other                     | `gaussian`, `clark`, `non_intersection`, `mean_censored_euclidean`, `statistic`              |

#### `general.py` — Evaluation & Utilities

| Function                    | Description                                                        |
|-----------------------------|--------------------------------------------------------------------|
| `confusion_matrix(Y, P)`   | Builds an N×N confusion matrix from true labels and predictions     |
| `normalize(array)`         | Z-score normalization (subtract mean, divide by std)                |
| `opf_accuracy(Y, P)`       | OPF-specific accuracy: `1 - Σ(type I + type II errors) / 2C`       |
| `opf_accuracy_per_label()` | Per-class accuracy breakdown                                        |
| `pre_compute_distance()`   | Generates and saves a full distance matrix to file                  |
| `purity(Y, P)`             | Clustering purity metric from confusion matrix                      |

#### `random.py` — Random Number Generation

| Function                             | Distribution |
|--------------------------------------|--------------|
| `generate_uniform_random_number()`   | Uniform      |
| `generate_gaussian_random_number()`  | Gaussian     |

---

### 3.3 `models` — Classifier Implementations

All four models inherit from `OPF` and follow the scikit-learn-inspired `fit()` / `predict()` pattern.

#### 3.3.1 `SupervisedOPF` (OPF → SupervisedOPF)

**Reference:** Papa, Falcão & Suzuki — *Supervised Pattern Classification based on Optimum-Path Forest* (2009).

**Algorithm (fit):**
1. Build a `Subgraph` from training data.
2. **Find prototypes** via Minimum Spanning Tree (MST): Run Prim's algorithm over all training nodes. Edges connecting nodes of *different classes* flag both endpoints as `PROTOTYPE`.
3. **Optimum-path competition:** Initialize prototypes with cost 0, all others with `FLOAT_MAX`. Using a **min-heap**, propagate labels outward. Each node joins the tree of the prototype that offers the lowest maximum-cost path (minimax path cost).

**Algorithm (predict):**
For each test sample, iterate over training nodes (sorted by cost) and find the training node that offers the minimum-cost connection. Assign that node's label. If a conqueror is found, mark its path as relevant.

**Additional capabilities:**
- `learn()` — Iterative learning: fit → predict → swap misclassified samples between train/val → repeat until convergence or max iterations.
- `prune()` — Iterative pruning: remove irrelevant nodes from the training set across iterations to reduce model size.

#### 3.3.2 `KNNSupervisedOPF` (OPF → KNNSupervisedOPF)

**Reference:** Papa & Falcão — *A Learning Algorithm for the Optimum-Path Forest Classifier* (2009).

**Key difference:** Uses a `KNNSubgraph` instead of a complete graph. Learns the optimal `k` value for the KNN adjacency relation over a validation set.

**Algorithm (fit):**
1. For each `k` from 1 to `max_k`:
   - Build KNN arcs, compute the probability density function (PDF), cluster the subgraph, predict on validation set, measure accuracy.
2. Select the `k` with highest accuracy.
3. Rebuild arcs and PDF with `best_k`, then cluster with `force_prototype=True` (ensures each class has at least one prototype).

**Algorithm (predict):**
For each test sample, find its `best_k` nearest training neighbours. Compute its density, scale it to the training distribution, then assign the label from the neighbour that maximizes `min(neighbour_cost, density)`.

#### 3.3.3 `SemiSupervisedOPF` (SupervisedOPF → SemiSupervisedOPF)

**Reference:** Amorim, Falcão & Carvalho — *Semi-supervised Pattern Classification Using Optimum-Path Forest* (2014).

**Algorithm (fit):**
1. Build a `Subgraph` from labeled training data and find prototypes (same MST approach as SupervisedOPF).
2. Append **unlabeled samples** as additional nodes (label = 0).
3. Run the same optimum-path competition. Unlabeled nodes inherit predicted labels from their conquerors. Additionally, `node.label` is set to `node.predicted_label` for consistency.

**Prediction** is inherited from `SupervisedOPF.predict()`.

#### 3.3.4 `UnsupervisedOPF` (OPF → UnsupervisedOPF)

**Reference:** Rocha, Cappabianco & Falcão — *Data clustering as an optimum-path forest problem* (2009).

**Algorithm (fit):**
1. Build a `KNNSubgraph` from training data.
2. **Best minimum cut:** For each `k` in `[min_k, max_k]`, create arcs, compute PDF, cluster, and evaluate using the **normalized cut** criterion. Select the `k` that minimizes the cut.
3. Perform final clustering with `best_k`.

**Clustering (`_clustering`):**
Uses a **max-heap**. Nodes with no predecessor become cluster roots (assigned unique `cluster_label`). Each node's cost propagates via `min(parent_cost, node_density)`. Nodes on density plateaus get bidirectional adjacency.

**Algorithm (predict):**
Same KNN-based prediction as `KNNSupervisedOPF`, but also propagates `cluster_label`.

**Additional:** `propagate_labels()` assigns the root node's ground-truth label to all samples in each cluster tree.

---

### 3.4 `stream` — Data I/O Pipeline

#### `loader.py` — File Loading

| Function       | Input Format | Parser        | Notes                                    |
|----------------|--------------|---------------|------------------------------------------|
| `load_csv()`   | `.csv`       | `np.loadtxt`  | Comma-delimited                           |
| `load_txt()`   | `.txt`       | `np.loadtxt`  | Space-delimited                           |
| `load_json()`  | `.json`      | `json.load`   | Expects `{"data": [{"id", "label", "features"}]}` |

All loaders return `np.ndarray` or `None` on failure.

#### `parser.py` — OPF Format Parsing

`parse_loader(data)` interprets the loaded NumPy array in **OPF file format**:
- **Column 0:** Sample index/id (discarded during parsing)
- **Column 1:** Class label
- **Columns 2+:** Feature values

Returns `(X, Y)` where `X` is the feature matrix and `Y` is the integer label array. Validates that labels are sequential starting from 0.

#### `splitter.py` — Data Splitting

| Function              | Returns                                      | Description                                      |
|-----------------------|----------------------------------------------|--------------------------------------------------|
| `split(X, Y, %)`     | `X_1, X_2, Y_1, Y_2`                         | Random permutation split                          |
| `split_with_index()`  | `X_1, X_2, Y_1, Y_2, I_1, I_2`              | Same, but also returns original index arrays       |
| `merge(X1, X2, Y1, Y2)` | `X, Y`                                     | Vertically stacks features, horizontally stacks labels |

---

### 3.5 `subgraphs` — Specialized Graph Structures

#### `KNNSubgraph` (Subgraph → KNNSubgraph)

Extends `Subgraph` with properties and methods needed for density-based and KNN-based OPF variants.

| Attribute/Method                | Description                                                                 |
|---------------------------------|-----------------------------------------------------------------------------|
| `n_clusters`                    | Number of discovered clusters                                                |
| `best_k`                        | Optimal k value for KNN adjacency                                           |
| `constant`                      | Scaling constant for PDF (`2 * density / 9`)                                |
| `density`, `min_density`, `max_density` | Density statistics across all nodes                               |
| `create_arcs(k, dist_fn, ...)`  | Builds k-nearest-neighbour adjacency lists using insertion sort              |
| `calculate_pdf(k, dist_fn, ...)`| Computes probability density per node, normalizes to `[1, MAX_DENSITY]`     |
| `eliminate_maxima_height(h)`    | Flattens density peaks above a threshold (for controlling cluster granularity)|

**Arc creation algorithm:** For each node, compute distances to all other nodes, maintain a sorted list of the `k` closest neighbours via insertion sort, and store them in `node.adjacency`. Also tracks `node.radius` (max distance among k-NN) and the global `density` (maximum arc distance).

---

### 3.6 `utils` — Cross-Cutting Utilities

#### `constants.py`

| Constant         | Value             | Usage                                               |
|------------------|-------------------|------------------------------------------------------|
| `EPSILON`        | `1e-20`           | Prevents division by zero in distance metrics         |
| `FLOAT_MAX`      | `sys.float_info.max` | Initial cost for unvisited nodes                   |
| `WHITE/GRAY/BLACK` | `0/1/2`        | Heap node colors (unseen / in-heap / processed)       |
| `NIL`            | `-1`              | No-predecessor sentinel                               |
| `STANDARD/PROTOTYPE` | `0/1`         | Node role flags                                       |
| `IRRELEVANT/RELEVANT` | `0/1`        | Node relevance flags (for pruning)                    |
| `MAX_ARC_WEIGHT` | `100000`          | Scaling factor for log-distance metrics               |
| `MAX_DENSITY`    | `1000`            | Upper bound for normalized density values             |

#### `converter.py` — Binary OPF Format Converters

Reads LibOPF's binary `.dat`/`.opf` file format (little-endian: 3-int header `[n_samples, n_labels, n_features]` followed by `[id, label, features...]` per sample) and converts to:
- `.txt` (space-delimited)
- `.csv` (comma-delimited)
- `.json` (`{"data": [...]}`)

Note: Labels are decremented by 1 during conversion (LibOPF uses 1-indexed labels).

#### `decorator.py`

`@avoid_zero_division` — A decorator that adds `EPSILON` to both input arrays before passing them to the wrapped distance function. Applied to ~25 distance metrics that involve division.

#### `exception.py`

Custom exception hierarchy rooted at `Error(Exception)`:

| Exception       | Purpose                                          |
|-----------------|--------------------------------------------------|
| `ArgumentError` | Wrong number of arguments                         |
| `BuildError`    | Object not properly initialized before use         |
| `SizeError`     | Mismatched array dimensions                        |
| `TypeError`     | Wrong variable type                                |
| `ValueError`    | Invalid variable value                             |

All exceptions log the error message via the package logger on construction.

#### `logging.py`

Factory function `get_logger(name)` returns a `Logger` with:
- **Console handler:** Streams to `stdout`.
- **Timed file handler:** Writes to `opfython.log` with midnight rotation.
- **Format:** `%(asctime)s - %(name)s — %(levelname)s — %(message)s`
- **Level:** `DEBUG` (all messages captured).

---

## 4. Class Hierarchy

```
Exception
└── Error                         (utils/exception.py)
    ├── ArgumentError
    ├── BuildError
    ├── SizeError
    ├── TypeError
    └── ValueError

Subgraph                          (core/subgraph.py)
└── KNNSubgraph                   (subgraphs/knn.py)

OPF                               (core/opf.py)
├── SupervisedOPF                 (models/supervised.py)
│   └── SemiSupervisedOPF         (models/semi_supervised.py)
├── KNNSupervisedOPF              (models/knn_supervised.py)
└── UnsupervisedOPF               (models/unsupervised.py)
```

---

## 5. Data Flow

```
                        ┌─────────────┐
                        │ Raw Files   │
                        │ .csv/.txt/  │
                        │ .json/.dat  │
                        └──────┬──────┘
                               │
                   ┌───────────▼───────────┐
                   │   stream/loader.py    │  load_csv() / load_txt() / load_json()
                   └───────────┬───────────┘
                               │ np.ndarray (raw)
                   ┌───────────▼───────────┐
                   │   stream/parser.py    │  parse_loader() → (X, Y)
                   └───────────┬───────────┘
                               │ X: features, Y: labels
                   ┌───────────▼───────────┐
                   │  stream/splitter.py   │  split() → X_train, X_test, Y_train, Y_test
                   └───────────┬───────────┘
                               │
              ┌────────────────▼────────────────┐
              │     Subgraph / KNNSubgraph      │  _build() creates Node objects
              │  (core/subgraph.py, subgraphs/) │
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │    OPF Model .fit(X, Y)         │  Heap-driven graph algorithms
              │  (models/*.py)                  │  using math/distance.py
              └────────────────┬────────────────┘
                               │ trained model
              ┌────────────────▼────────────────┐
              │    OPF Model .predict(X)        │  Returns List[int] predictions
              └────────────────┬────────────────┘
                               │
              ┌────────────────▼────────────────┐
              │   math/general.py               │  opf_accuracy(), purity(), etc.
              └─────────────────────────────────┘
```

---

## 6. Distance Metrics Catalog

The default distance metric is **`log_squared_euclidean`**, defined as:

```
d(x,y) = MAX_ARC_WEIGHT × log(Σ(xᵢ - yᵢ)² + 1)
```

All 44 metrics are registered in `math.distance.DISTANCES` and can be selected by string name at classifier instantiation:

```python
opf = SupervisedOPF(distance="euclidean")
```

---

## 7. Key Algorithms

### 7.1 Prototype Discovery (SupervisedOPF)
Uses **Prim's MST algorithm** over the complete graph. After building the MST, any edge connecting nodes of different classes identifies both endpoints as **prototypes** (decision-boundary representatives).

### 7.2 Optimum-Path Competition (SupervisedOPF, SemiSupervisedOPF)
A modified **Dijkstra's algorithm** with a **minimax path cost** function:
- Prototypes start with cost 0; all others with ∞.
- Cost of a path = maximum arc weight along the path.
- Each node is conquered by the prototype offering the lowest maximum-cost path.
- **Min-heap** drives the exploration order.

### 7.3 Density-Based Clustering (KNNSupervisedOPF, UnsupervisedOPF)
- Build KNN graph, compute PDF (Gaussian kernel over k-NN distances).
- Normalize densities to `[1, MAX_DENSITY]`.
- Run **max-heap** competition: nodes propagate via `min(parent_cost, node_density)`.
- Each unrooted node (no predecessor) becomes a cluster root.

### 7.4 Normalized Cut (UnsupervisedOPF)
After clustering, compute:
```
cut = Σ_l [external(l) / (internal(l) + external(l))]
```
Where `internal(l)` sums reciprocal distances within cluster `l`, and `external(l)` sums reciprocal distances to other clusters. The `k` minimizing this cut is selected.

---

## 8. Serialization & Persistence

- **Model persistence:** `OPF.save(path)` / `OPF.load(path)` use Python `pickle`.
- **Distance matrix caching:** `general.pre_compute_distance()` saves a distance matrix via `np.savetxt()`.
- **Binary format conversion:** `converter.opf2txt()`, `opf2csv()`, `opf2json()` convert LibOPF binary files.

---

## 9. Dependencies

| Package      | Min Version | Role                                        |
|--------------|-------------|---------------------------------------------|
| `numpy`      | ≥ 1.19.5   | Array operations, linear algebra             |
| `numba`      | ≥ 0.53.0   | JIT compilation of distance functions        |
| `coverage`   | ≥ 5.5      | Test coverage reporting                      |
| `pytest`     | ≥ 6.2.2    | Test framework                               |
| `pylint`     | ≥ 2.7.2    | Static analysis                              |
| `pre-commit` | ≥ 2.17.0   | Git hook management                          |

**Runtime-critical:** Only `numpy` and `numba` are needed for inference. The remaining are development dependencies.
