Usage and contracts
===================

OPFython keeps model configuration separate from graph state. Choose the model
family and metric, fit it with the required datasets, and predict only after
training completes. Refit after changing metric configuration.

.. testsetup::

    import logging

    _logging_threshold = logging.root.manager.disable
    logging.disable(logging.CRITICAL)

Supervised classification
-------------------------

Feature matrices have one sample per row. Labels contain one nonnegative
integer per sample. Class identifiers are zero-based and sequential for a
complete dataset, although an evaluation split can omit some classes.

.. testcode::

    import numpy as np

    from opfython.models import SupervisedOPF

    features = np.asarray([[0.0, 0.0], [0.1, 0.2], [1.0, 1.0], [1.1, 0.9]])
    labels = np.asarray([0, 0, 1, 1])
    queries = np.asarray([[0.05, 0.1], [1.05, 1.0]])

    model = SupervisedOPF()
    result = model.fit(features, labels)
    predictions = model.predict(queries)

    assert result is None
    assert predictions == [0, 1]

The supervised, KNN-supervised, and semi-supervised prediction methods return a
list of class labels. Unsupervised prediction returns a pair containing the
class-label list and the cluster-assignment list. Unsupervised class labels
are populated by the separate ``propagate_labels()`` operation.

KNN-supervised fitting also requires a validation feature/label pair to select
the neighbourhood size. Semi-supervised fitting instead requires an unlabeled
feature matrix. KNN model fitting requires ``1 <= k < n_training_samples``.

Input ownership
---------------

Graph nodes can retain views of the supplied feature arrays. Do not modify
training features behind a fitted model without refitting it. Built-in distance
guards prepare their own adjusted values rather than writing to caller storage.

.. testcode::

    from opfython.math.distance import canberra_distance

    left = np.asarray([0.0, 1.0])
    right = np.asarray([1.0, 0.0])
    left.flags.writeable = False

    assert canberra_distance(left, right) == 2.0
    np.testing.assert_array_equal(left, [0.0, 1.0])
    np.testing.assert_array_equal(right, [1.0, 0.0])
    assert not left.flags.writeable

``SupervisedOPF.learn()`` intentionally exchanges samples between the supplied
training and validation arrays in place. It uses the application's NumPy random
state, so seed that state explicitly when reproducible learning is required.
Supervised prediction records relevant paths in the model for pruning.

Splitting and original indexes
------------------------------

The splitter's ``random_state`` controls only the split. It preserves the
established seeded permutation without reseeding NumPy's application-wide
generator. ``split_with_index`` returns features, labels, and original row
indexes in that order.

.. testcode::

    from opfython.stream.splitter import split_with_index

    rows = np.arange(12).reshape(6, 2)
    row_labels = np.arange(6)
    first, second, first_labels, second_labels, first_ids, second_ids = split_with_index(
        rows, row_labels, percentage=0.5, random_state=1
    )

    np.testing.assert_array_equal(first_ids, [2, 1, 4])
    np.testing.assert_array_equal(second_ids, [0, 3, 5])
    np.testing.assert_array_equal(first, rows[first_ids])
    np.testing.assert_array_equal(second, rows[second_ids])

When using precomputed distances, pass the corresponding original indexes to
``fit`` and ``predict``. Matrix dimensions must cover the row and column indexes
actually accessed, not merely match the size of a training subset.

Semi-supervised unlabeled samples use consecutive matrix indexes starting at
``len(X_train)``. Arrange the matrix accordingly, as the current API has no
separate unlabeled-index argument.

Loading data
------------

OPF dataset rows contain an identifier, a class label, and then feature values.
``parse_loader`` returns only features and labels, discarding the identifier
column. Supply matrix indexes separately when constructing an indexed graph.

CSV and text loaders log file-access failures and return ``None``. JSON loading
also retains its documented read/decode failure sentinels. Other malformed-data
errors propagate. Inspect the function contracts rather than treating every
failure as an empty dataset.

Persistence and logging
-----------------------

``save`` writes the current model state as a pickle, and ``load`` updates an
existing instance in place. Use the saved model's class when restoring it.
Only load trusted files, as pickle can execute code. I/O and serialization
errors propagate to the caller.

.. testcode::

    from pathlib import Path
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        path = Path(directory) / "model.pkl"
        model.save(path)
        restored = SupervisedOPF()
        restored.load(path)
        assert restored.predict(queries) == predictions

Compatibility depends on the model code and numerical dependencies. A
same-version round trip is not a promise that arbitrary future releases can
load every older pickle.

An unconfigured package logger uses stdout and a delayed, midnight-rotating
``opfython.log`` file. Repeated ``get_logger`` calls must not duplicate handlers
or overwrite application-owned configuration. Applications control their
logging policy through Python's logging facilities.

.. testcleanup::

    logging.disable(_logging_threshold)
