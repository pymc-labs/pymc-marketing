import numpy as np
import pandas as pd
import pymc as pm
import pytest
from lifetimes import BetaGeoFitter as BGF

from pymc_marketing.clv.models.beta_geo_beta_binom import BetaGeoBetaBinomModel


@pytest.fixture(scope="module")
def test_donations() -> pd.DataFrame:
    """
    Load donations benchmark dataset into a Pandas dataframe.
    This dataset aggregates identical customers by count,
    and should be exploded into one customer per row for testing.

    Data source: https://www.brucehardie.com/datasets/
    """

    count_df = pd.read_csv("datasets/donations.csv")

    agg_df = count_df.drop("count", axis=1)

    for row in zip(agg_df.values, count_df["count"]):
        array = np.tile(row[0], (row[1], 1))
        try:
            concat_array = np.concatenate((concat_array, array), axis=0)
        except NameError:
            concat_array = array

    exploded_df = pd.DataFrame(concat_array, columns=["frequency", "recency", "T"])

    assert len(exploded_df) == count_df["count"].sum()

    return exploded_df
