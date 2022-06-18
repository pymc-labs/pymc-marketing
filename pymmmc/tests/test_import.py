def test_import_pymc():
    import pymc as pm

    assert pm.__version__ == "4.0.0"
