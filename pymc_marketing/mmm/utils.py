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
from pymc.logprob.basic import logcdf, logp
from pytensor import xtensor as ptx
from pytensor.graph.basic import Variable
from pytensor.graph.replace import graph_replace


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
) -> xr.Dataset:
    """Create an ``xr.Dataset`` for future prediction, with zero fills.

    Creates a dataset with dates from *start_date* to *end_date* and all model
    dimensions, filling channel and control variables with zeros (or with values
    from *channel_xr* if provided).  The output has the canonical underscore
    variable names (``_channel``, ``_control``).

    Parameters
    ----------
    model
        Fitted MMM instance.  Must have ``xarray_dataset``, ``date_column``,
        ``channel_columns``, ``control_columns``, ``dims`` and ``adstock``
        attributes.
    start_date, end_date
        Date range for the prediction period.
    channel_xr
        Optional per-dimension channel values.  Data variables must be a subset
        of ``model.channel_columns``.  Dimensions must be a subset of
        ``model.dims`` and must **not** include the date dimension.  Values are
        broadcast across every date in the generated range.
    include_carryover
        Whether to extend the date range by ``adstock.l_max`` periods so that
        adstock initialisation has enough leading observations.

    Returns
    -------
    xr.Dataset
        Dataset with ``_channel`` (and optionally ``_control``) variables,
        indexed by ``("date", *dims, "channel")``.
    """
    if not hasattr(model, "xarray_dataset"):
        raise ValueError(
            "Model must have a fitted 'xarray_dataset'. "
            "Call `build_model` / `fit` first."
        )

    if not hasattr(model, "date_column"):
        raise ValueError("Model must expose a `.date_column` attribute.")

    required_attrs = ("channel_columns", "control_columns", "dims", "adstock")
    for attr in required_attrs:
        if not hasattr(model, attr):
            raise ValueError(f"Model must have a '{attr}' attribute.")

    xa = model.xarray_dataset
    channel_cols = list(model.channel_columns)
    control_cols = (
        list(model.control_columns) if model.control_columns is not None else []
    )
    dim_cols = list(model.dims)

    # ---- 1. Infer date frequency from training data ---------------------------
    training_dates = pd.DatetimeIndex(xa.coords["date"].values)
    inferred_freq = pd.infer_freq(training_dates)
    if inferred_freq is None:
        warnings.warn(
            "Could not infer frequency from training dates. Using weekly ('W').",
            UserWarning,
            stacklevel=2,
        )
        inferred_freq = "W"

    # ---- 2. Build date range --------------------------------------------------
    if include_carryover:
        if not isinstance(start_date, pd.Timestamp):
            start_date = pd.Timestamp(start_date)
        if not isinstance(end_date, pd.Timestamp):
            end_date = pd.Timestamp(end_date)
        if hasattr(model.adstock, "l_max"):
            end_date += _convert_frequency_to_timedelta(
                model.adstock.l_max, inferred_freq
            )

    new_dates = pd.date_range(start=start_date, end=end_date, freq=inferred_freq)
    if new_dates.empty:
        raise ValueError("Generated date range is empty. Check dates and frequency.")

    n_dates = len(new_dates)

    # ---- 3. Dimension coordinates from training data --------------------------
    dim_coords = {}
    for dim in dim_cols:
        dim_coords[dim] = xa.coords[dim].values

    # ---- 4. Build _channel variable -------------------------------------------
    chan_shape = [n_dates]
    chan_coords: dict = {"date": new_dates}
    for dim in dim_cols:
        chan_shape.append(len(dim_coords[dim]))
        chan_coords[dim] = dim_coords[dim]
    chan_shape.append(len(channel_cols))
    chan_coords["channel"] = channel_cols

    channel_data = np.zeros(chan_shape, dtype=float)

    # ---- 4a. Inject channel_xr values -----------------------------------------
    if channel_xr is not None:
        if isinstance(channel_xr, xr.DataArray):
            channel_name = channel_xr.name or "value"
            channel_xr = channel_xr.to_dataset(name=channel_name)

        if not isinstance(channel_xr, xr.Dataset):
            raise TypeError(
                "`channel_xr` must be an xarray Dataset or DataArray, "
                f"got {type(channel_xr).__name__}."
            )

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

        if "date" in channel_xr.dims:
            raise ValueError("`channel_xr` must NOT include the date dimension.")

        for ch in channel_cols:
            if ch in channel_xr.data_vars:
                ch_idx = channel_cols.index(ch)
                vals = channel_xr[ch].values
                channel_data[..., ch_idx] = np.broadcast_to(
                    vals, (n_dates, *vals.shape)
                )

    # ---- 5. Build _control variable -------------------------------------------
    data_vars: dict = {
        "_channel": xr.DataArray(
            channel_data, dims=("date", *dim_cols, "channel"), coords=chan_coords
        ),
    }

    if control_cols:
        ctrl_shape = [n_dates]
        ctrl_coords: dict = {"date": new_dates}
        for dim in dim_cols:
            ctrl_shape.append(len(dim_coords[dim]))
            ctrl_coords[dim] = dim_coords[dim]
        ctrl_shape.append(len(control_cols))
        ctrl_coords["control"] = control_cols

        data_vars["_control"] = xr.DataArray(
            np.zeros(ctrl_shape, dtype=float),
            dims=("date", *dim_cols, "control"),
            coords=ctrl_coords,
        )

    return xr.Dataset(data_vars)


def add_noise_to_channel_allocation(
    df: pd.DataFrame | xr.Dataset,
    channels: list[str],
    rel_std: float = 0.05,
    seed: int | None = None,
) -> pd.DataFrame | xr.Dataset:
    """Add Gaussian noise to channel values.

    Accepts both ``pd.DataFrame`` (with *channels* as columns) and
    ``xr.Dataset`` (with a ``_channel`` data variable).  The return type
    matches the input type.
    """
    rng = np.random.default_rng(seed)

    if isinstance(df, xr.Dataset):
        da = df["_channel"]
        non_channel_dims = tuple(d for d in da.dims if d != "channel")
        ch_scale = (rel_std * da.mean(dim=non_channel_dims)).values
        noise = rng.normal(loc=0.0, scale=ch_scale, size=da.shape)
        noisy = xr.where(da == 0, 0.0, da + noise).clip(min=0.0)
        result = df.copy()
        result["_channel"] = noisy
        return result

    scale: np.ndarray = (rel_std * df[channels].mean()).to_numpy()
    noise = rng.normal(loc=0.0, scale=scale, size=(len(df), len(channels)))
    noisy_df = df.copy()
    noisy_df[channels] += noise
    zero_spend_mask = df[channels] == 0
    noisy_df[zero_spend_mask] = 0.0
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
    idata : xr.DataTree-like
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


def density(dist, *, value, **params: Variable):
    """Request density of dist at value.

    This helper creates dist with dummy `params`, requests its density from `pymc.logp`
    and then  reintroduces the original `params` values.
    This avoids accidental rewrite of random graphs above params when
    the logp cannot be obtained by direct dispatch
    """
    masked_params = {k: p.type() for k, p in params.items()}
    masked_dist = dist.dist(**masked_params)
    masked_density = ptx.math.exp(logp(masked_dist, value))
    return graph_replace(
        masked_density,
        tuple(zip(masked_params.values(), params.values(), strict=True)),
        strict=False,
    )


def cdf(dist, *, value, **params: Variable):
    """Request CDF of dist at value.

    This helper creates dist with dummy `params`, requests its cdf from `pymc.logcdf`
    and then  reintroduces the original `params` values.
    This avoids accidental rewrite of random graphs above params when
    the logcdf cannot be obtained by direct dispatch
    """
    masked_params = {k: p.type() for k, p in params.items()}
    masked_dist = dist.dist(**masked_params)
    masked_cdf = ptx.math.exp(logcdf(masked_dist, value))
    return graph_replace(
        masked_cdf,
        tuple(zip(masked_params.values(), params.values(), strict=True)),
        strict=False,
    )
