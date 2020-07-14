from opfython.stream import loader


def test_load_csv():
    csv = loader.load_csv('boat.csv')

    assert csv is None

    csv = loader.load_csv('data/boat.csv')

    assert csv.shape == (100, 4)


def test_load_txt():
    txt = loader.load_txt('boat.txt')

    assert txt is None

    txt = loader.load_txt('data/boat.txt')

    assert txt.shape == (100, 4)


def test_load_json():
    json = loader.load_json('boat.json')

    assert json is None

    json = loader.load_json('data/boat.json')

    assert json.shape == (100, 4)
