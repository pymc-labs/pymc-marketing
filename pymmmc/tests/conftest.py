import pandas as pd
import pytest

from pymmmc.datasets import cdnow_summary, cdnow_transactions, donations


@pytest.fixture(scope="module")
def cdnow_rfm() -> pd.DataFrame:
    """Create a test dataset from the CDNOW RFM summary for use in multiple tests."""

    return cdnow_summary()


@pytest.fixture(scope="module")
def cdnow_trans() -> pd.DataFrame:
    """Create a test dataset of CDNOW transactions for use in multiple tests."""

    return cdnow_transactions()


@pytest.fixture(scope="module")
def donations_rfm() -> pd.DataFrame:
    """Create a test dataset from the Donations summary for use in multiple tests."""

    return donations()
