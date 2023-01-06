import importlib_resources
import numpy as np
import pandas as pd

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

    ref = importlib_resources.files("pymc_marketing") / "datasets" / "cdnow_summary.csv"
    with importlib_resources.as_file(ref) as path:
        return pd.read_csv(path, **kwargs)


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

    ref = importlib_resources.files("pymc_marketing") / "datasets" / "CDNOW_sample.txt"
    with importlib_resources.as_file(ref) as path:
        return pd.read_csv(
            path,
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

    ref = importlib_resources.files("pymc_marketing") / "datasets" / "donations.csv"
    with importlib_resources.as_file(ref) as path:
        donations = pd.read_csv(path, **kwargs)

    donations_df = donations.drop("count", axis=1)

    for row in zip(donations_df.values, donations["count"]):
        array = np.tile(row[0], (row[1], 1))
        try:
            concat_array = np.concatenate((concat_array, array), axis=0)
        except NameError:
            concat_array = array

    return pd.DataFrame(concat_array, columns=["frequency", "recency", "periods"])
