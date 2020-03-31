import os

import numpy as np
import pytest

from opfython.core import opf
from opfython.core.subgraph import Subgraph


def test_opf_subgraph():
    clf = opf.OPF()

    try:
        clf.subgraph = 'a'
    except:
        clf.subgraph = Subgraph()

    assert isinstance(clf.subgraph, Subgraph)


def test_opf_distance():
    clf = opf.OPF()

    try:
        clf.distance = 'a'
    except:
        clf.distance = 'euclidean'

    assert clf.distance == 'euclidean'


def test_opf_distance_fn():
    clf = opf.OPF()

    try:
        clf.distance_fn = 'a'
    except:
        clf.distance_fn = callable

    assert clf.distance_fn == callable


def test_opf_pre_computed_distance():
    clf = opf.OPF()

    try:
        clf.pre_computed_distance = 'a'
    except:
        clf.pre_computed_distance = False

    assert clf.pre_computed_distance == False


def test_opf_pre_distances():
    clf = opf.OPF()

    try:
        clf.pre_distances = 'a'
    except:
        clf.pre_distances = np.ones(10)

    assert clf.pre_distances.shape == (10,)


def test_opf_read_distances():
    try:
        clf = opf.OPF(pre_computed_distance='data/boat')
    except:
        clf = opf.OPF(pre_computed_distance='data/boat.txt')
    
    assert clf.pre_distances.shape == (100, 4)

    try:
        clf = opf.OPF(pre_computed_distance='data/boa.txt')
    except:
        clf = opf.OPF(pre_computed_distance='data/boat.csv')

    assert clf.pre_distances.shape == (100, 4)


def test_opf_save():
    clf = opf.OPF(distance='bray_curtis')

    clf.save('data/test.pkl')

    assert os.path.isfile('data/test.pkl')


def test_opf_load():
    clf = opf.OPF()

    clf.load('data/test.pkl')

    assert clf.distance == 'bray_curtis'


def test_opf_fit():
    clf = opf.OPF()

    with pytest.raises(NotImplementedError):
        clf.fit(None, None)


def test_opf_predict():
    clf = opf.OPF()

    with pytest.raises(NotImplementedError):
        clf.predict(None)
