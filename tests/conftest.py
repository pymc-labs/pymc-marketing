import pandas as pd
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="module")
def cdnow_trans() -> pd.DataFrame:
    """
    Load CDNOW_sample transaction data into a Pandas dataframe.

    Data source: https://www.brucehardie.com/datasets/
    """
    return pd.read_csv("tests/clv/datasets/cdnow_transactions.csv")
