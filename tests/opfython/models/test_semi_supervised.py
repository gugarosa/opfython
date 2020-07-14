import numpy as np

from opfython.models import semi_supervised
from opfython.stream import loader, parser

csv = loader.load_csv('data/boat.csv')
X, Y = parser.parse_loader(csv)


def test_supervised_opf_fit():
    opf = semi_supervised.SemiSupervisedOPF()

    opf.fit(X, Y, X)

    opf.pre_computed_distance = True

    try:
        opf.pre_distances = np.ones((100, 100))
        opf.fit(X, Y, X)
    except:
        opf.pre_distances = np.ones((200, 200))
        opf.fit(X, Y, X)

    assert opf.subgraph.trained == True
