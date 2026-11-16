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
"""Utilities for the CLV module."""

from collections.abc import Sequence

import numpy as np
import pandas
import xarray

# Re-exported: the RFM functions lived in this module before they moved to ``rfm``.
from pymc_marketing.clv.rfm import (  # noqa: F401
    rfm_segments,
    rfm_summary,
    rfm_train_test_split,
)
from pymc_marketing.constants import DAYS_IN_MONTH, DAYS_IN_WEEK
from pymc_marketing.names.clv import Col, Dim

__all__ = ["customer_lifetime_value", "to_xarray"]

# The units ``customer_lifetime_value`` computes with are pandas period aliases.
# pandas 3 removed "H" ("h" replaces it), and the named units are the ones the RFM
# functions take, so that one spelling works through a whole analysis.
_PERIOD_ALIASES = {"H": "h", "day": "D", "week": "W", "month": "M"}


def _normalize_time_unit(time_unit: str) -> str:
    """Return the pandas period alias for a public ``time_unit`` value."""
    return _PERIOD_ALIASES.get(time_unit, time_unit)


def _resolve_dims(ndim: int, dims: tuple[str, ...]) -> tuple[str, ...]:
    """Return the dim names labelling an array of ``ndim`` dimensions."""
    if ndim == 1:
        return dims[:1]
    if ndim == len(dims):
        return dims
    if ndim == 0:
        raise ValueError(
            f"Cannot convert a 0-dimensional array: every array needs a leading "
            f"{dims[0]!r} axis."
        )
    if len(dims) == 1:
        raise ValueError(
            f"Cannot label an array of {ndim} dimensions with {dims}. Pass "
            "'extra_dims' naming its trailing axes."
        )
    raise ValueError(
        f"Cannot label an array of {ndim} dimensions with {dims}: every array must "
        f"have either 1 or {len(dims)} dimensions. Arrays whose trailing axes differ "
        "need one call each, so that every axis is named unambiguously."
    )


def to_xarray(
    customer_id,
    *arrays,
    dim: str = "customer_id",
    extra_dims: Sequence[str] | None = None,
):
    """Convert arrays to xarray objects sharing a common dim (default "customer_id").

    The leading axis of every array is labelled ``dim`` and takes ``customer_id`` as
    its coordinate. Arrays with more than one axis are supported, but the names of
    their trailing axes must be supplied through ``extra_dims``. Auto-generated names
    are deliberately not provided: they would make the unrelated trailing axes of two
    arrays returned by the same call align with each other.

    For the same reason every array must be one-dimensional or carry the full set of
    trailing axes; a rank in between is rejected rather than guessed at, since
    ``extra_dims`` would not say which of its names the missing axis dropped. Arrays
    with different trailing axes need one call each.

    Parameters
    ----------
    customer_id : array-like
        Coordinate values for ``dim``. Must match the length of the leading axis of
        every array.
    *arrays : array-like
        Arrays to convert. Each must be one-dimensional or have
        ``1 + len(extra_dims)`` dimensions.
    dim : str, default "customer_id"
        Name of the leading dimension.
    extra_dims : sequence of str, optional
        Names of the trailing dimensions, e.g. ``("channel",)``. Required as soon as
        an array is multidimensional. A bare string is rejected: it would be read as
        one name per character.

    Returns
    -------
    xarray.DataArray or tuple of xarray.DataArray
        One ``DataArray`` per entry in ``arrays``, or a bare ``DataArray`` when a
        single array is passed.

    Raises
    ------
    TypeError
        If ``extra_dims`` is a string rather than a sequence of names.
    ValueError
        If ``extra_dims`` repeats a name or clashes with ``dim``, or if an array is
        neither one-dimensional nor of dimension ``1 + len(extra_dims)``.

    Examples
    --------
    .. code-block:: python

        frequency, recency = to_xarray(
            data["customer_id"], data["frequency"], data["recency"]
        )
        spend = to_xarray(data["customer_id"], channel_matrix, extra_dims=("channel",))

    Trailing dimensions are named but not populated; attach their labels with
    ``assign_coords``:

    .. code-block:: python

        spend = to_xarray(
            data["customer_id"], channel_df, extra_dims=("channel",)
        ).assign_coords(channel=channel_df.columns)

    """
    if isinstance(extra_dims, str):
        raise TypeError(
            "extra_dims must be a sequence of dimension names, not a string; pass "
            f"extra_dims=({extra_dims!r},) to name a single trailing axis."
        )

    dims: tuple[str, ...] = (dim, *(extra_dims or ()))
    if len(set(dims)) != len(dims):
        raise ValueError(f"Dimension names must be unique, got {dims}.")

    coords = {dim: np.asarray(customer_id)}

    res = tuple(
        xarray.DataArray(
            data=array, coords=coords, dims=_resolve_dims(np.ndim(array), dims)
        )
        for array in arrays
    )

    return res[0] if len(arrays) == 1 else res


def customer_lifetime_value(
    transaction_model,
    data: pandas.DataFrame,
    future_t: int = 12,
    discount_rate: float = 0.00,
    time_unit: str = "D",
) -> xarray.DataArray:
    """
    Compute customer lifetime value.

    Compute the average lifetime value for a group of one or more customers
    and apply a discount rate for net present value estimations.

    Note: ``future_t`` is always in months regardless of the ``time_unit``
    used for the transaction model. The value is converted internally to
    the given ``time_unit`` for computing expected purchases per period.

    Adapted from the legacy ``lifetimes`` library:
    https://github.com/CamDavidsonPilon/lifetimes/blob/41e394923ad72b17b5da93e88cfabab43f51abe2/lifetimes/utils.py#L449

    Parameters
    ----------
    transaction_model : ~CLVModel
        Predictive model for future transactions. `BetaGeoModel`,
        `ModifiedBetaGeoModel`, and `ParetoNBDModel` are currently supported.
    data : ~pandas.DataFrame
        DataFrame containing the following columns:

        * `customer_id`: Unique customer identifier
        * `frequency`: Number of repeat purchases observed for each customer
        * `recency`: Time between the first and the last purchase
        * `T`: Time between the first purchase and the end of the observation period
        * `future_spend`: Predicted monetary values for each customer
    future_t : int, optional
        The number of months to project lifetime value for. This is always
        specified in months, independent of ``time_unit``. Default: 12
    discount_rate : float, optional
        The monthly adjusted discount rate. Default: 0.00
    time_unit : string, optional
        Unit of time of the purchase history. Defaults to "D" for daily.
        Other options are "W" (weekly), "M" (monthly), and "H" (hourly).
        The names the RFM functions take, "day", "week" and "month", work too.
        Example: If your dataset contains information about weekly purchases,
        you should use "W".

    Returns
    -------
    xarray
        DataArray containing estimated customer lifetime values

    """
    time_unit = _normalize_time_unit(time_unit)
    if Col.FUTURE_SPEND not in data.columns:
        raise ValueError("Required column future_spend missing")

    def _squeeze_dims(x: xarray.DataArray):
        """
        Squeeze dimensions for MAP-fitted model predictions.

        This utility is required for MAP-fitted model predictions to broadcast properly.

        Parameters
        ----------
        x : xarray.DataArray
            DataArray to squeeze dimensions for.

        Returns
        -------
        xarray.DataArray
            DataArray with squeezed dimensions.
        """
        dims_to_squeeze: tuple[str, ...] = ()
        if "chain" in x.dims and len(x.chain) == 1:
            dims_to_squeeze += ("chain",)
        if "draw" in x.dims and len(x.draw) == 1:
            dims_to_squeeze += ("draw",)
        x = x.squeeze(dims_to_squeeze)
        return x

    if discount_rate == 0.0:
        # no discount rate: just compute a single time step from 0 to `time`
        steps = np.arange(future_t, future_t + 1)
    else:
        steps = np.arange(1, future_t + 1)

    # DAYS_IN_MONTH is the average calendar month (365.25 / 12). A flat 30 days
    # understates a year by 5.25 days and inflates CLV for the "D", "W" and "h" units.
    factor = {
        "W": DAYS_IN_MONTH / DAYS_IN_WEEK,
        "M": 1.0,
        "D": DAYS_IN_MONTH,
        "h": DAYS_IN_MONTH * 24,
    }[time_unit]

    monetary_value = to_xarray(data[Col.CUSTOMER_ID], data[Col.FUTURE_SPEND])

    clv = xarray.DataArray(0.0)

    # initialize FOR loop with 0 purchases at future_t = 0
    prev_expected_purchases = 0

    for i in steps * factor:
        # since the prediction of number of transactions is cumulative, we have to subtract off the previous periods
        new_expected_purchases = _squeeze_dims(
            transaction_model.expected_purchases(
                data=data,
                future_t=i,
            )
        )
        expected_transactions = new_expected_purchases - prev_expected_purchases
        prev_expected_purchases = new_expected_purchases

        # sum up the CLV estimates of all the periods and apply discounted cash flow
        clv = clv + (monetary_value * expected_transactions) / (1 + discount_rate) ** (
            i / factor
        )

    # Add squeezed chain/draw dims
    if "draw" not in clv.dims:
        clv = clv.expand_dims({"draw": 1})
    if "chain" not in clv.dims:
        clv = clv.expand_dims({"chain": 1})

    return clv.transpose("chain", "draw", Dim.CUSTOMER_ID)
