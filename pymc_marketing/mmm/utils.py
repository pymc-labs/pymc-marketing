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

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
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


def create_zero_dataset(
    model: Any,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Create a zero-filled dataset for model prediction over a specified date range.

    Creates a DataFrame for prediction with zero values for channel and control columns,
    preserving the original data's date frequency.

    Parameters
    ----------
    model : MMM
        An instance of the pymc_marketing MMM class.
    start_date : str or pd.Timestamp
        The start date for the new DataFrame (inclusive).
    end_date : str or pd.Timestamp
        The end date for the new DataFrame (inclusive).

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame structured for prediction, with zeros in channel
        and control columns, and matching the inferred date frequency.

    Raises
    ------
    ValueError
        If essential attributes are missing from the model object
        or if the date column cannot be processed.
    """
    if not hasattr(model, "X") or not isinstance(model.X, pd.DataFrame):
        raise ValueError(
            "MMM object must have an 'X' attribute containing a pandas DataFrame."
        )
    if not hasattr(model, "date_column") or model.date_column not in model.X.columns:
        raise ValueError(
            "MMM object must have a valid 'date_column' attribute corresponding to a column in mmm.X."
        )
    if not hasattr(model, "channel_columns"):
        raise ValueError("MMM object must have a 'channel_columns' attribute.")
    if not hasattr(model, "control_columns"):
        raise ValueError("MMM object must have a 'control_columns' attribute.")
    if not hasattr(model, "dims"):
        raise ValueError("MMM object must have a 'dims' attribute.")

    original_data = model.X
    date_col = model.date_column
    channel_cols = model.channel_columns
    control_cols = model.control_columns
    dim_cols = list(model.dims)  # Ensure it's a list

    # --- Frequency Inference ---
    try:
        # Ensure date column is datetime type
        date_series = pd.to_datetime(original_data[date_col])
        # Infer frequency from unique sorted dates
        inferred_freq = pd.infer_freq(date_series.unique())
        if inferred_freq is None:
            warnings.warn(
                f"Could not infer date frequency from column '{date_col}'. "
                "Defaulting to daily frequency ('D'). Check if dates are regular.",
                UserWarning,
                stacklevel=2,
            )
            inferred_freq = "D"  # Default to daily if inference fails
        else:
            print(f"Inferred date frequency: {inferred_freq}")
    except Exception as e:
        warnings.warn(
            f"Error during frequency inference for column '{date_col}': {e}. "
            "Defaulting to daily frequency ('D').",
            UserWarning,
            stacklevel=2,
        )
        inferred_freq = "D"
    # --- End Frequency Inference ---

    # 1. Generate Date Range using inferred frequency
    try:
        new_dates = pd.date_range(
            start=start_date,
            end=end_date,
            freq=inferred_freq,  # Use inferred frequency
            name=date_col,
        )
        if new_dates.empty:
            raise ValueError(
                "Date range resulted in empty dates. Check start/end dates and frequency."
            )
    except ValueError as e:
        raise ValueError(
            f"""Error creating date range: {e}. Ensure start_date and end_date are valid and compatible with
            inferred frequency '{inferred_freq}'."""
        ) from e

    date_df = pd.DataFrame(new_dates)

    # 2. Get Unique Dimension Combinations
    if dim_cols:
        unique_dims = original_data[dim_cols].drop_duplicates().reset_index(drop=True)
        # 3. Cross Join Dates and Dimensions
        # Add temporary keys for cross join
        date_df["_key"] = 1
        unique_dims["_key"] = 1
        pred_df = pd.merge(date_df, unique_dims, on="_key").drop("_key", axis=1)
    else:
        # If no dims, the prediction frame just has the date column
        pred_df = date_df

    # 4. Add Channel and Control Columns with Zeros
    for col in channel_cols + control_cols:
        if col not in pred_df.columns:  # Avoid overwriting dim cols if they overlap
            pred_df[col] = 0.0

    # 5. Add any other columns present in original_data, filling with 0
    other_cols = [
        col
        for col in original_data.columns
        if col not in (*[date_col], *dim_cols, *channel_cols, *control_cols)
    ]
    for col in other_cols:
        if col not in pred_df.columns:
            pred_df[col] = 0.0

    # 6. Ensure correct column order
    final_columns = original_data.columns
    # Handle cases where a dim column might also be a channel/control (unlikely but possible)
    pred_df = pred_df[[col for col in final_columns if col in pred_df.columns]]
    # Add any missing columns (e.g., if original had only dates, no dims/channels/controls)
    for col in final_columns:
        if col not in pred_df.columns:
            # Determine appropriate fill value (0 for numeric, maybe mode/NaN for categoricals not in dims?)
            # Sticking to 0.0 for simplicity as per original request goal.
            pred_df[col] = 0.0

    # Reapply final desired column order
    pred_df = pred_df[final_columns]

    # 7. Ensure correct data types (optional but good practice)
    for col in final_columns:
        if col in original_data.columns and col in pred_df.columns:
            target_dtype = original_data[col].dtype
            try:
                pred_df[col] = pred_df[col].astype(target_dtype)
            except Exception as e:
                warnings.warn(
                    f"Could not cast column '{col}' to original dtype {target_dtype}: {e}",
                    UserWarning,
                    stacklevel=2,
                )

    return pred_df
