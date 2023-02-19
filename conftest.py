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
def cdnow_data() -> pd.DataFrame:
    """
    Load CDNOW_sample.txt into a Pandas dataframe for model and utility function testing.
    """
    df = pd.read_csv(
        "tests/clv/datasets/CDNOW_sample.txt",
        sep=r"\s+",
        header=None,
        names=["_id", "id", "date", "cds_bought", "spent"],
    )
    return df
