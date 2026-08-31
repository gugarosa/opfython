import numpy as np

from opfython.models import SemiSupervisedOPF
from opfython.stream import loader, parser

X, Y = parser.parse_loader(loader.load_csv("data/boat.csv"))


def test_semi_supervised_fit():
    classifier = SemiSupervisedOPF()

    classifier.fit(X, Y, X)

    assert classifier.subgraph.trained is True


def test_semi_supervised_uses_precomputed_distances():
    classifier = SemiSupervisedOPF()
    classifier.pre_computed_distance = True
    classifier.pre_distances = np.ones((200, 200))

    classifier.fit(X, Y, X)

    assert classifier.subgraph.trained is True
