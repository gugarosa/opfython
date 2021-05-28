from opfython.math import general
from opfython.stream import loader, parser, splitter


def test_confusion_matrix():
    labels = [0, 0, 1, 1]
    preds = [0, 0, 1, 1]

    c_matrix = general.confusion_matrix(labels, preds)

    assert c_matrix.shape == (2, 2)


def test_normalize():
    array = [1, 1, 1, 2]

    norm_array = general.normalize(array)

    assert norm_array[3] == 1.7320508075688774


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


def test_opf_pre_compute_distances():
    txt = loader.load_txt('data/boat.txt')

    X, Y = parser.parse_loader(txt)

    X_train, _, _, _ = splitter.split(X, Y, 0.5, 1)

    general.pre_compute_distance(X_train, 'boat_split_distances.txt', 'log_squared_euclidean')


def test_purity():
    labels = [0, 0, 1, 1]
    preds = [0, 0, 1, 1]

    purity = general.purity(labels, preds)

    assert purity == 1
