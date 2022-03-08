import pymc as pm


def test_import_pymc():
    assert pm.__version__ == "4.0.0b2"


def test_dummy():
    assert True
