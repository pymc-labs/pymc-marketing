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
"""Reference arithmetic the period readouts are checked against.

``eval_mask`` gives the boolean mask the reduction read before it read slices.
``split_band`` is how the current/future split summed a period's joined band
before :class:`~pymc_marketing.mmm.period_readout.PeriodCurrentFuture` reduced
it block by block.  It lives here, not in the package, so the readout and its
reference cannot drift into the same code.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pandas.tseries.offsets import BaseOffset

from pymc_marketing.mmm.counterfactual import PeriodWindow
from pymc_marketing.mmm.period_readout import current_window_end


def split_band(
    band: xr.DataArray,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lags: int,
    freq_offset: BaseOffset,
) -> tuple[xr.DataArray, xr.DataArray, bool]:
    """Reduce one period's band to its current and future value.

    Current is every realization date from the period's start through its
    end plus ``lags`` data periods, future every date after that.

    Parameters
    ----------
    band : xr.DataArray
        The period's per-date increment on ``realization_date``, with a
        boolean ``observed`` coordinate on the same dimension.  Unobserved
        dates hold ``NaN``.
    start, end : pd.Timestamp
        The period's bounds.
    lags : int
        Data periods after ``end`` that still count as current.
    freq_offset : BaseOffset
        The data's date frequency, which ``lags`` is counted in.

    Returns
    -------
    current, future : xr.DataArray
        *band* summed over each window, without ``realization_date``.
        Neither sum skips ``NaN``: an unobserved date is carryover the model
        has and the data cannot show, not a zero.
    complete : bool
        Whether every date in the band is observed.
    """
    current_end = current_window_end(end, lags=lags, freq_offset=freq_offset)
    realization = band.indexes["realization_date"]
    in_current = (realization >= start) & (realization <= current_end)

    current = band.isel(realization_date=in_current).sum(
        "realization_date", skipna=False
    )
    future = band.isel(realization_date=realization > current_end).sum(
        "realization_date", skipna=False
    )
    complete = bool(band.coords["observed"].all())

    return current, future, complete


def eval_mask(window: PeriodWindow, max_window: int) -> np.ndarray:
    """Positions in the padded window that enter the sum, as a boolean mask.

    The masks the reduction read before it read slices; the tests check that
    the slices select the same dates.
    """
    mask = np.zeros(max_window, dtype=bool)
    mask[np.flatnonzero(window.in_eval[window.in_window])] = True
    return mask
