#   Copyright 2022 - 2026 The PyMC Labs Developers
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


def _convert_frequency_to_timedelta(periods: int, freq: str) -> pd.Timedelta:
    """Convert frequency string and periods to Timedelta.

    Parameters
    ----------
    periods : int
        Number of periods
    freq : str
        Frequency string (e.g., 'D', 'W', 'M', 'Y')

    Returns
    -------
    pd.Timedelta
        The timedelta representation
    """
    # Extract base frequency (e.g., 'W' from 'W-MON')
    base_freq = freq[0] if len(freq) > 1 else freq

    # Direct mapping for supported frequencies
    if base_freq == "D":
        return pd.Timedelta(days=periods)
    elif base_freq == "W":
        return pd.Timedelta(weeks=periods)
    elif base_freq == "M":
        # Approximate months as 30 days
        return pd.Timedelta(days=periods * 30)
    elif base_freq == "Y":
        # Approximate years as 365 days
        return pd.Timedelta(days=periods * 365)
    elif base_freq == "H":
        return pd.Timedelta(hours=periods)
    elif base_freq == "T":
        return pd.Timedelta(minutes=periods)
    elif base_freq == "S":
        return pd.Timedelta(seconds=periods)
    else:
        # Default to weeks if frequency not recognized
        warnings.warn(
            f"Unrecognized frequency '{freq}'. Defaulting to weeks.",
            UserWarning,
            stacklevel=2,
        )
        return pd.Timedelta(weeks=periods)


def create_zero_dataset(
    model: Any,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    channel_xr: xr.Dataset | xr.DataArray | None = None,
    include_carryover: bool = True,
) -> pd.DataFrame:
    """Create a DataFrame for future prediction, with zeros (or supplied constants).

    Creates a DataFrame with dates from start_date to end_date and all model dimensions,
    filling channel and control columns with zeros or with values from channel_xr if provided.

    If *channel_xr* is provided it must

    • have data variables that are a subset of ``model.channel_columns``
    • be indexed *only* by the dimensions in ``model.dims`` (no date dimension)

    The values in *channel_xr* are copied verbatim to the corresponding channel
    columns and broadcast across every date in ``start_date … end_date``.
    """
    # ---- 0. Basic integrity checks (unchanged) --------------------------------
    if not hasattr(model, "X") or not isinstance(model.X, pd.DataFrame):
        raise ValueError("'model.X' must be a pandas DataFrame.")

    if not hasattr(model, "date_column") or model.date_column not in model.X.columns:
        raise ValueError(
            "Model must expose `.date_column` and that column must be in `model.X`."
        )

    required_attrs = ("channel_columns", "control_columns", "dims")
    for attr in required_attrs:
        if not hasattr(model, attr):
            raise ValueError(f"Model must have a '{attr}' attribute.")

    original_data = model.X
    date_col = model.date_column
    channel_cols = list(model.channel_columns)
    control_cols = (
        list(model.control_columns) if model.control_columns is not None else []
    )
    dim_cols = list(model.dims)  # ensure list

    # ---- 1. Infer date frequency ------------------------------------------------
    date_series = pd.to_datetime(original_data[date_col])
    inferred_freq = pd.infer_freq(date_series.unique())
    if inferred_freq is None:  # fall-back if inference fails
        warnings.warn(
            f"Could not infer frequency from '{date_col}'. Using weekly ('W').",
            UserWarning,
            stacklevel=2,
        )
        inferred_freq = "W"

    # ---- 2. Build the full Cartesian product of dates X dims -------------------
    if include_carryover:
        # if start_date are not timestamps, convert them to timestamps
        if not isinstance(start_date, pd.Timestamp):
            start_date = pd.Timestamp(start_date)
        if not isinstance(end_date, pd.Timestamp):
            end_date = pd.Timestamp(end_date)

        # Add the adstock lag to the end date
        if hasattr(model.adstock, "l_max"):
            end_date += _convert_frequency_to_timedelta(
                model.adstock.l_max, inferred_freq
            )

    new_dates = pd.date_range(
        start=start_date, end=end_date, freq=inferred_freq, name=date_col
    )
    if new_dates.empty:
        raise ValueError("Generated date range is empty. Check dates and frequency.")

    date_df = pd.DataFrame(new_dates)

    if dim_cols:  # cross-join with dimension levels
        unique_dims = original_data[dim_cols].drop_duplicates().reset_index(drop=True)
        date_df["_k"] = 1
        unique_dims["_k"] = 1
        pred_df = pd.merge(date_df, unique_dims, on="_k").drop(columns="_k")
    else:
        pred_df = date_df.copy()

    # ---- 3. Initialise channel & control columns with zeros --------------------
    for col in channel_cols + control_cols:
        if col not in pred_df.columns:  # don't overwrite dim columns by accident
            pred_df[col] = 0.0

    # ---- 4. Optional channel_xr injection --------------------------------------
    if channel_xr is not None:
        # --- 4.1 Normalise to Dataset ------------------------------------------
        if isinstance(channel_xr, xr.DataArray):
            # Give the single DataArray a name equal to its channel (attr 'name')
            channel_name = channel_xr.name or "value"
            channel_xr = channel_xr.to_dataset(name=channel_name)

        if not isinstance(channel_xr, xr.Dataset):
            raise TypeError("`channel_xr` must be an xarray Dataset or DataArray.")

        # --- 4.2 Validate variables & dimensions -------------------------------
        invalid_vars = set(channel_xr.data_vars) - set(channel_cols)
        if invalid_vars:
            raise ValueError(
                f"`channel_xr` contains variables not in `model.channel_columns`: "
                f"{sorted(invalid_vars)}"
            )

        missing_channels = set(channel_cols) - set(channel_xr.data_vars)
        if missing_channels:
            warnings.warn(
                f"`channel_xr` does not supply values for {sorted(missing_channels)}; "
                "they will stay at 0.",
                UserWarning,
                stacklevel=2,
            )

        invalid_dims = set(channel_xr.dims) - set(dim_cols)
        if invalid_dims:
            raise ValueError(
                f"`channel_xr` uses dims that are not recognised model dims: "
                f"{sorted(invalid_dims)}"
            )

        if date_col in channel_xr.dims:
            raise ValueError("`channel_xr` must NOT include the date dimension.")

        # --- 4.3 Inject constants ----------------------------------------------
        # Special-case: when there are NO dims (e.g., only channel dimension in the
        # allocation which was pivoted into variables), xarray can't create an index
        # for to_dataframe(). In this scenario, simply broadcast scalar values
        # across all rows.
        if len(channel_xr.dims) == 0:
            for ch in channel_cols:
                if ch in channel_xr.data_vars:
                    # assign scalar value across all rows
                    try:
                        pred_df[ch] = channel_xr[ch].item()
                    except Exception:
                        pred_df[ch] = channel_xr[ch].values
        else:
            # Convert to DataFrame & merge when dims are present
            channel_df = channel_xr.to_dataframe().reset_index()

            # Left-join on every dimension; suffix prevents collisions during merge
            pred_df = pred_df.merge(
                channel_df,
                on=dim_cols,
                how="left",
                suffixes=("", "_chan"),
            )

        # --- 4.4 Copy merged values into official channel columns --------------
        if len(channel_xr.dims) != 0:
            for ch in channel_cols:
                chan_col = f"{ch}_chan"
                if chan_col in pred_df.columns:
                    pred_df[ch] = pred_df[chan_col]
                    pred_df.drop(columns=chan_col, inplace=True)

            # Replace any remaining NaNs introduced by the merge
            pred_df[channel_cols] = pred_df[channel_cols].fillna(0.0)

    # ---- 5. Bring in any “other” columns from the training data ----------------
    other_cols = [
        col
        for col in original_data.columns
        if col not in [date_col, *dim_cols, *channel_cols, *control_cols]
    ]
    for col in other_cols:
        if col not in pred_df.columns:
            pred_df[col] = 0.0

    # ---- 6. Match original column order & dtypes ------------------------------
    final_columns = original_data.columns
    pred_df = pred_df.reindex(columns=final_columns)

    for col in final_columns:
        try:
            pred_df[col] = pred_df[col].astype(original_data[col].dtype)
        except Exception as e:
            warnings.warn(
                f"Could not cast '{col}' to {original_data[col].dtype}: {e}",
                UserWarning,
                stacklevel=2,
            )

    return pred_df


def add_noise_to_channel_allocation(
    df: pd.DataFrame,
    channels: list[str],
    rel_std: float = 0.05,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Return *df* with additive Gaussian noise applied to *channels* columns.

    Parameters
    ----------
    df : DataFrame
        The original data (will **not** be modified in-place).
    channels : list of str
        Column names whose values represent media spends.
    rel_std : float, default 0.05
        Noise standard-deviation expressed as a fraction of the
        *column mean* (i.e. `0.05` ⇒ 5 % of the mean spend).
    seed : int or None
        Optional seed for deterministic output.

    Returns
    -------
    DataFrame
        A copy of *df* with noisy spends.
    """
    rng = np.random.default_rng(seed)

    # Per-channel scale (1-D ndarray), shape (n_channels,)
    scale: np.ndarray = (rel_std * df[channels].mean()).to_numpy()

    # Draw all required noise in one call, shape (n_rows, n_channels)
    noise = rng.normal(loc=0.0, scale=scale, size=(len(df), len(channels)))

    # Create the noisy copy
    noisy_df = df.copy()
    noisy_df[channels] += noise

    # Override channels with zero spend, we don't want to add noise to those ones
    zero_spend_mask = df[channels] == 0
    noisy_df[zero_spend_mask] = 0.0

    # Ensure no negative spends
    noisy_df[channels] = noisy_df[channels].clip(lower=0.0)

    return noisy_df


def create_index(
    dims: tuple[str, ...],
    take: tuple[str, ...],
) -> tuple[int | slice, ...]:
    """Create an index to take the first dimension of a tensor based on the provided dimensions."""
    return tuple(slice(None) if dim in take else 0 for dim in dims)


def build_contributions(
    idata,
    var: list[str] | tuple[str, ...],
    agg: str | Callable = "mean",
    *,
    agg_dims: list[str] | tuple[str, ...] | None = None,
    index_dims: list[str] | tuple[str, ...] | None = None,
    expand_dims: list[str] | tuple[str, ...] | None = None,
    cast_regular_to_category: bool = True,
) -> pd.DataFrame:
    """Build a wide contributions DataFrame from idata.posterior variables.

    This function extracts contribution variables from the posterior,
    aggregates them across sampling dimensions, and returns a wide DataFrame
    with automatic dimension detection and handling.

    Parameters
    ----------
    idata : az.InferenceData-like
        Must have `.posterior` attribute containing the contribution variables.
    var : list or tuple of str
        Posterior variable names to include (e.g., contribution variables).
    agg : str or callable, default "mean"
        xarray reduction method applied over `agg_dims` for each variable.
        Can be "mean", "median", "sum", or any callable reduction function.
    agg_dims : list or tuple of str, optional
        Sampling dimensions to reduce over. If None, defaults to
        ("chain", "draw") but only includes dimensions that exist.
    index_dims : list or tuple of str, optional
        Dimensions to preserve as index-like columns. If None, defaults
        to ("date",) but only includes dimensions that exist.
    expand_dims : list or tuple of str, optional
        Dimensions whose coordinates should become separate wide columns.
        If None, defaults to ("channel", "control"). Only one such dimension
        is expected per variable.
    cast_regular_to_category : bool, default True
        Whether to cast non-index regular dimensions to pandas 'category' dtype.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with columns for:
        - Index dimensions (e.g., date)
        - Regular dimensions (e.g., geo, product)
        - One column per label in each expand dimension (e.g., channel__C1, control__x1)
        - Single columns for scalar variables (e.g., intercept)

    Raises
    ------
    ValueError
        If none of the requested variables are present in idata.posterior.

    Examples
    --------
    Build contributions DataFrame with default settings:

    .. code-block:: python

        df = build_contributions(
            idata=mmm.idata,
            var=[
                "intercept_contribution_original_scale",
                "channel_contribution_original_scale",
                "control_contribution_original_scale",
            ],
        )

    Use median aggregation instead of mean:

    .. code-block:: python

        df = build_contributions(
            idata=mmm.idata,
            var=["channel_contribution"],
            agg="median",
        )

    """
    # Set defaults for dimension handling
    if agg_dims is None:
        agg_dims = ("chain", "draw")
    if index_dims is None:
        index_dims = ("date",)
    if expand_dims is None:
        expand_dims = ("channel", "control")

    # Select and validate variables
    present = [v for v in var if v in idata.posterior]
    if not present:
        raise ValueError(
            f"None of the requested variables {var} are present in idata.posterior."
        )

    def _reduce(da: xr.DataArray) -> xr.DataArray:
        """Reduce DataArray over aggregation dimensions."""
        dims = tuple(d for d in agg_dims if d in da.dims)
        if not dims:
            return da
        if isinstance(agg, str):
            return getattr(da, agg)(dim=dims)
        return da.reduce(agg, dim=dims)

    # Reduce each variable
    reduced = {v: _reduce(idata.posterior[v]) for v in present}

    # Discover union of "regular" dims and their coords
    special = set(expand_dims) | set(agg_dims) | {"variable"}
    all_dims = set().union(*(set(da.dims) for da in reduced.values()))
    regular_dims = [d for d in all_dims if d not in special]

    # Collect union coordinates (keep index_dims order first)
    coord_unions = {}
    for d in set(regular_dims) | set(index_dims):
        idxs = [
            pd.Index(da.coords[d].to_pandas())
            for da in reduced.values()
            if d in da.dims
        ]
        if not idxs:
            continue
        u = idxs[0]
        for idx in idxs[1:]:
            u = u.union(idx)
        coord_unions[d] = u

    # Create template grid for broadcasting
    template = xr.DataArray(0)
    for d, idx in coord_unions.items():
        template = template.expand_dims({d: idx})

    # Expand variables with channel/control dimension, broadcast others
    datasets = []
    for name, da in reduced.items():
        da_b = xr.broadcast(da, template)[0] if template.dims else da

        # Detect expand dimension (at most one expected per variable)
        exp_dim = next((d for d in expand_dims if d in da_b.dims), None)
        if exp_dim is not None:
            # Convert to dataset with wide columns: "<exp_dim>__<label>"
            ds = da_b.to_dataset(dim=exp_dim)
            ds = ds.rename({v: f"{exp_dim}__{v}" for v in ds.data_vars})
            datasets.append(ds)
        else:
            short_name = name.removesuffix("_original_scale").removesuffix(
                "_contribution"
            )
            datasets.append(da_b.to_dataset(name=short_name))

    # Merge all datasets
    ds_all = (
        xr.merge(datasets, compat="override", join="outer")
        if len(datasets) > 1
        else datasets[0]
    )

    # Stable column order: index_dims first, then other regular dims
    ordered_dims = [d for d in index_dims if d in ds_all.dims] + [
        d for d in regular_dims if d not in index_dims and d in ds_all.dims
    ]

    df = ds_all.to_dataframe().reset_index()

    # Cast non-index regular dims to category (memory & modeling friendly)
    if cast_regular_to_category:
        for d in ordered_dims:
            if d not in index_dims and d in df:
                df[d] = df[d].astype("category")

    # Sort for readability
    sort_cols = [c for c in ordered_dims if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable")

    return df
