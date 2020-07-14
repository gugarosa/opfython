import os

from opfython.utils import converter


def test_opf2txt():
    converter.opf2txt('data/boat.dat')

    assert os.path.isfile('data/boat.txt')


def test_opf2csv():
    converter.opf2csv('data/boat.dat')

    assert os.path.isfile('data/boat.csv')


def test_opf2json():
    converter.opf2json('data/boat.dat')

    assert os.path.isfile('data/boat.json')
