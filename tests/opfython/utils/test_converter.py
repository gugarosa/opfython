import os

import pytest
from opfython.utils import converter


def test_opf2txt():
    converter.opf2txt('tests/opfython/data/boat.dat')

    assert os.path.isfile('tests/opfython/data/boat.txt')


def test_opf2csv():
    converter.opf2csv('tests/opfython/data/boat.dat')

    assert os.path.isfile('tests/opfython/data/boat.csv')


def test_opf2json():
    converter.opf2json('tests/opfython/data/boat.dat')

    assert os.path.isfile('tests/opfython/data/boat.json')
