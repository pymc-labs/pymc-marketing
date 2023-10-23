from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

from pymc_marketing.mmm.utils import generate_yearly_fourier_modes

__all__ = ["FourierTransformer", "create_mmm_transformer"]


class FourierTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_order: int) -> None:
        self.n_order = n_order

    def fit(self, X: pd.Series) -> "FourierTransformer":
        dummy_value = np.array([0])
        self.columns = generate_yearly_fourier_modes(
            dayofyear=dummy_value, n_order=self.n_order
        ).columns
        return self

    def transform(self, X: pd.Series) -> pd.DataFrame:
        return generate_yearly_fourier_modes(X.dt.dayofyear, n_order=self.n_order)

    def get_feature_names_out(self):
        return self.columns.to_numpy()


def create_mmm_transformer(
    channel_cols: List[str],
    date_col: Optional[str] = None,
    yearly_fourier_order: Optional[int] = None,
    control_cols: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Create the default transformer for the MMM model that will be used in the class

    Parameters
    ----------
    channel_cols : List[str]
        The columns that contain the channel data.
    date_col : Optional[str]
        The column that contains the date data.
    yearly_fourier_order : Optional[int]
        The order of the Fourier series to use for the yearly seasonality.
    control_cols : Optional[List[str]]
        The columns that contain the control data.

    Returns
    -------
    ColumnTransformer
        The transformer that will be used for the MMM model.

    """

    transformers = [("channel", MaxAbsScaler(), channel_cols)]

    if control_cols is not None:
        transformers.append(("control", StandardScaler(), control_cols))

    if date_col is not None and yearly_fourier_order is not None:
        transformers.append(
            ("fourier_mode", FourierTransformer(n_order=yearly_fourier_order), date_col)  # type: ignore
        )

    return ColumnTransformer(transformers, verbose_feature_names_out=False).set_output(
        transform="pandas"
    )
