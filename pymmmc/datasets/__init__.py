import numpy as np
import pandas as pd
from pkg_resources import resource_filename

__all__ = [
    "cdnow_summary",
    "cdnow_transactions",
    "donations",
]


def cdnow_summary(**kwargs) -> pd.DataFrame:
    """
    Load the CDNOW RFM summary dataset into a Pandas DataFrame.

    This is a benchmarking dataset for continuous, noncontractual transactions.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments passed into pandas.read_csv function.

    Returns
    -------
    DataFrame
    """

    return pd.read_csv(
        resource_filename("pymmmc", "datasets/cdnow_summary.csv"), **kwargs
    )


def cdnow_transactions(**kwargs) -> pd.DataFrame:
    """
    Load the CDNOW raw transactions dataset into a Pandas DataFrame.

    This is a benchmarking dataset for continuous, noncontractual transactions.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments passed into pandas.read_csv function.

    Returns
    -------
    DataFrame
    """

    return pd.read_csv(
        resource_filename("pymmmc", "datasets/CDNOW_sample.txt"),
        sep=r"\s+",
        header=None,
        names=["master_id", "sample_id", "date", "cds_bought", "spent"],
        **kwargs
    )


def donations(**kwargs) -> pd.DataFrame:
    """
    Load the Donations RFM summary dataset into a Pandas DataFrame.

    This is a benchmarking dataset for discrete, noncontractual transactions.

    Parameters
    ----------
    **kwargs
        Additional keyword arguments passed into pandas.read_csv function.

    Returns
    -------
    DataFrame
    """

    donations = pd.read_csv(
        resource_filename("pymmmc", "datasets/donations.csv"), **kwargs
    )

    donations_df = donations.drop("count", axis=1)

    for row in zip(donations_df.values, donations["count"]):
        array = np.tile(row[0], (row[1], 1))
        try:
            concat_array = np.concatenate((concat_array, array), axis=0)
        except NameError:
            concat_array = array

    return pd.DataFrame(concat_array, columns=["frequency", "recency", "periods"])
