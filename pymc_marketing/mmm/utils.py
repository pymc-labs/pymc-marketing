#   Copyright 2022 - 2025 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Utility functions for the Marketing Mix Modeling module."""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr


def apply_sklearn_transformer_across_dim(
    data: xr.DataArray,
    func: Callable[[np.ndarray], np.ndarray],
    dim_name: str,
) -> xr.DataArray:
    """Apply a scikit-learn transformer across a dimension of an xarray DataArray.

    Helper function in order to use scikit-learn functions with the xarray target.

    Parameters
    ----------
    data : xr.DataArray
        The input data to transform.
    func : Callable[[np.ndarray], np.ndarray]
        scikit-learn method to apply to the data
    dim_name : str
        Name of the dimension to apply the function to

    Returns
    -------
    xr.DataArray

    """
    # These are lost during the ufunc
    attrs = data.attrs
    # Cache dims to restore them after the ufunc
    dims = data.dims

    data = (
        xr.apply_ufunc(
            func,
            data.expand_dims("_"),
            input_core_dims=[[dim_name, "_"]],
            output_core_dims=[[dim_name, "_"]],
            vectorize=True,
            on_missing_core_dim="copy",
        )
        .squeeze(dim="_")
        .transpose(*dims)
    )

    data.attrs = attrs

    return data


def transform_1d_array(
    transform: Callable[[pd.Series | np.ndarray], np.ndarray], y: pd.Series | np.ndarray
) -> np.ndarray:
    """Transform a 1D array using a scikit-learn transformer.

    Parameters
    ----------
    transform : scikit-learn transformer
        The transformer to apply to the data.
    y : np.ndarray
        The data to transform.

    Returns
    -------
    np.ndarray
        The transformed data.

    """
    return transform(np.array(y)[:, None]).flatten()


def sigmoid_saturation(
    x: float | np.ndarray | npt.NDArray,
    alpha: float | np.ndarray | npt.NDArray,
    lam: float | np.ndarray | npt.NDArray,
) -> float | Any:
    """Sigmoid saturation function.

    Parameters
    ----------
    x : float or np.ndarray
        The input value for which the function is to be computed.
    alpha : float or np.ndarray
        α (alpha): Represent the Asymptotic Maximum or Ceiling Value.
    lam : float or np.ndarray
        λ (lambda): affects how quickly the function approaches its upper and lower asymptotes. A higher value of
        lam makes the curve steeper, while a lower value makes it more gradual.

    """
    if alpha <= 0 or lam <= 0:
        raise ValueError("alpha and lam must be greater than 0")

    return (alpha - alpha * np.exp(-lam * x)) / (1 + np.exp(-lam * x))


def create_new_spend_data(
    spend: np.ndarray,
    adstock_max_lag: int,
    one_time: bool,
    spend_leading_up: np.ndarray | None = None,
) -> np.ndarray:
    """Create new spend data for the channel forward pass.

    Spends must be the same length as the number of channels.

    .. plot::
        :context: close-figs

        import numpy as np
        import matplotlib.pyplot as plt
        import arviz as az

        from pymc_marketing.mmm.utils import create_new_spend_data
        az.style.use("arviz-white")

        spend = np.array([1, 2])
        adstock_max_lag = 3
        one_time = True
        spend_leading_up = np.array([4, 3])
        channel_spend = create_new_spend_data(spend, adstock_max_lag, one_time, spend_leading_up)

        time_since_spend = np.arange(-adstock_max_lag, adstock_max_lag + 1)

        ax = plt.subplot()
        ax.plot(
            time_since_spend,
            channel_spend,
            "o",
            label=["Channel 1", "Channel 2"]
        )
        ax.legend()
        ax.set(
            xticks=time_since_spend,
            yticks=np.arange(0, channel_spend.max() + 1),
            xlabel="Time since spend",
            ylabel="Spend",
            title="One time spend with spends leading up",
        )
        plt.show()


    Parameters
    ----------
    spend : np.ndarray
        The spend data for the channels.
    adstock_max_lag : int
        The maximum lag for the adstock transformation.
    one_time: bool, optional
        If the spend is one-time, by default True.
    spend_leading_up : np.ndarray, optional
        The spend leading up to the first observation, by default None or 0.

    Returns
    -------
    np.ndarray
        The new spend data for the channel forward pass.

    """
    n_channels = len(spend)

    if spend_leading_up is None:
        spend_leading_up = np.zeros_like(spend)

    if len(spend_leading_up) != n_channels:
        raise ValueError("spend_leading_up must be the same length as the spend")

    spend_leading_up = np.tile(spend_leading_up, adstock_max_lag).reshape(
        adstock_max_lag, -1
    )

    spend = (
        np.vstack([spend, np.zeros((adstock_max_lag, n_channels))])
        if one_time
        else np.ones((adstock_max_lag + 1, n_channels)) * spend
    )

    return np.vstack(
        [
            spend_leading_up,
            spend,
        ]
    )
