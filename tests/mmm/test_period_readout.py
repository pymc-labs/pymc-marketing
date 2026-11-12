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
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.mmm.counterfactual import EvaluationWindows
from pymc_marketing.mmm.incrementality import (
    IdentityLinkReducer,
    Incrementality,
    LogLinkReducer,
)
from pymc_marketing.mmm.period_readout import (
    LinearPredictorDelta,
    PeriodBand,
    PeriodCurrentFuture,
    PeriodTotal,
    ReadoutContext,
    append_unobserved_tail,
    split_band,
)

DATES = pd.date_range("2023-01-01", periods=6, freq="W-SUN")
WEEKLY = pd.tseries.frequencies.to_offset("W-SUN")


class TestLinearPredictorDelta:
    """The change in the linear predictor, differenced before anything is summed.

    Synthetic arrays, no model.  The reference is the arithmetic the change was
    computed with before it read through slices: every node's whole difference
    over the evaluation mask, then a channel column or the channel sum.
    """

    N_SAMPLES = 3
    GEOS = ["g0", "g1"]
    CHANNELS = ["c0", "c1", "c2"]
    # The channel axis is not the first non-date axis, so reading a column has
    # to find it.
    NON_DATE_DIMS = {
        "channel_contribution": ("geo", "channel"),
        "effect_contribution": ("geo",),
        "mu": ("geo",),
    }

    @pytest.fixture
    def windows(self):
        return EvaluationWindows.build(
            periods=[(DATES[0], DATES[1]), (DATES[3], DATES[4])],
            dates=DATES,
            l_max=2,
            freq_offset=WEEKLY,
        )

    @pytest.fixture
    def arrays(self, windows):
        rng = np.random.default_rng(0)
        sizes = {"geo": len(self.GEOS), "channel": len(self.CHANNELS)}
        n_rows = 2 * len(windows)
        counterfactual, baseline = {}, {}
        for name, dims in self.NON_DATE_DIMS.items():
            shape = tuple(sizes[dim] for dim in dims)
            counterfactual[name] = rng.normal(
                size=(n_rows, self.N_SAMPLES, windows.max_window, *shape)
            )
            baseline[name] = rng.normal(size=(self.N_SAMPLES, len(DATES), *shape))
        return counterfactual, baseline

    def _reference(self, counterfactual, baseline, row, mask, window, channel, effects):

        def delta(name):
            dims = self.NON_DATE_DIMS[name]
            return xr.DataArray(
                counterfactual[name][row][:, mask] - baseline[name][:, window.in_eval],
                dims=("sample", "date", *dims),
                coords={
                    "date": window.eval_dates,
                    "geo": self.GEOS,
                    **({"channel": self.CHANNELS} if "channel" in dims else {}),
                },
            )

        cc = delta("channel_contribution")
        result = (
            cc.sum(dim="channel")
            if channel is None
            else cc.isel(channel=channel, drop=True)
        )
        for name in effects:
            result = result + delta(name)
        return result

    @pytest.mark.parametrize("effects", [(), ("effect_contribution",)])
    @pytest.mark.parametrize("channel", [None, 0, 2])
    def test_matches_differencing_every_node_through_the_masks(
        self, windows, arrays, channel, effects
    ):
        """Bitwise the same change, column read or channels summed, with or without an effect."""
        counterfactual, baseline = arrays
        delta_mu = LinearPredictorDelta(
            effect_names=effects,
            non_date_dims=self.NON_DATE_DIMS,
            coords={"geo": np.array(self.GEOS), "channel": np.array(self.CHANNELS)},
        )

        for period_idx, window in enumerate(windows.windows):
            row = 2 * period_idx + 1
            got = delta_mu(
                counterfactual, baseline, row=row, window=window, channel=channel
            )

            xr.testing.assert_identical(
                got,
                self._reference(
                    counterfactual,
                    baseline,
                    row,
                    window.eval_mask(windows.max_window),
                    window,
                    channel,
                    effects,
                ),
            )


def _increment(values, dates, *, draws=1):
    """One period of one block: ``(chain, draw, realization_date, channel, geo)``.

    *values* are per date; every draw, channel and geo gets them scaled by its
    own factor, so a mix-up between entries would show.
    """
    values = np.asarray(values, dtype=float)
    factors = np.arange(1, draws * 2 * 2 + 1, dtype=float).reshape(1, draws, 1, 2, 2)
    return xr.DataArray(
        values.reshape(1, 1, -1, 1, 1) * factors,
        dims=("chain", "draw", "realization_date", "channel", "geo"),
        coords={
            "chain": [0],
            "draw": np.arange(draws),
            "realization_date": pd.DatetimeIndex(dates),
            "channel": ["c0", "c1"],
            "geo": ["g0", "g1"],
        },
    )


class TestPeriodReadouts:
    """Each readout keeps what its output reads, and only that.

    Synthetic bands, no model.  The reference for the current and future split
    is what the split computed before it had a readout: the band joined over
    every block, extended by its unobserved tail and put in the reported
    order, then :func:`split_band`.
    """

    # The reported order of ``geo`` differs from the order the increment
    # arrives in, as it can for a reloaded model.
    CONTEXT = ReadoutContext(
        dates=DATES,
        reported_order={"geo": np.array(["g1", "g0"])},
        tail_lags=3,
        freq_offset=WEEKLY,
    )

    @pytest.fixture
    def windows(self):
        # Week 1 lands inside the data; week 4 lands on weeks 4 to 7, and the
        # data ends at week 5, so weeks 6 and 7 are an unobserved tail.
        return EvaluationWindows.build(
            periods=[(DATES[1], DATES[1]), (DATES[4], DATES[4])],
            dates=DATES,
            l_max=3,
            freq_offset=WEEKLY,
        )

    def _reference(self, band, window, lags):
        joined = append_unobserved_tail(
            band,
            window=window,
            dates=DATES,
            tail_lags=self.CONTEXT.tail_lags,
            freq_offset=WEEKLY,
        ).sel(self.CONTEXT.reported_order)
        return split_band(
            joined, start=window.start, end=window.end, lags=lags, freq_offset=WEEKLY
        )

    def _read(self, readout, band, window, *, n_blocks):
        """Reduce *band* in *n_blocks* blocks of draws, join them, and finish."""
        edges = np.linspace(0, band.sizes["draw"], n_blocks + 1).astype(int)
        parts = [
            (
                {"chain": slice(0, 1), "draw": slice(start, stop)},
                readout.reduce_block(
                    band.isel(draw=slice(start, stop)),
                    window=window,
                    context=self.CONTEXT,
                ),
            )
            for start, stop in pairwise(edges)
        ]
        return readout.finish_period(
            Incrementality._join_sample_blocks(parts),
            window=window,
            context=self.CONTEXT,
        )

    @pytest.mark.parametrize("n_blocks", [1, 2])
    @pytest.mark.parametrize("lags", [0, 1, 2, 3])
    @pytest.mark.parametrize("period", [0, 1])
    def test_current_future_matches_splitting_the_joined_band(
        self, windows, period, lags, n_blocks
    ):
        """The same sums, ``NaN`` where an unobserved date falls, and ``complete``.

        For the period near the end: with ``lags=0`` the tail is future, so
        future is ``NaN`` and current is not; with ``lags=2`` the tail
        straddles both; with ``lags=3`` it is all current and future sums
        nothing.
        """
        window = windows.windows[period]
        band = _increment(
            np.arange(1.0, len(window.eval_dates) + 1), window.eval_dates, draws=4
        )
        expected_current, expected_future, expected_complete = self._reference(
            band, window, lags
        )

        current, future, complete = self._read(
            PeriodCurrentFuture(lags), band, window, n_blocks=n_blocks
        )

        xr.testing.assert_identical(current, expected_current.rename("current"))
        xr.testing.assert_identical(future, expected_future.rename("future"))
        assert complete is expected_complete
        assert complete is (period == 0)

    def test_the_tail_makes_a_sum_nan(self, windows):
        """An unobserved date is not summed as zero, on whichever side it falls."""
        window = windows.windows[1]
        band = _increment([1.0, 2.0], window.eval_dates)

        current, future, _ = self._read(
            PeriodCurrentFuture(0), band, window, n_blocks=1
        )
        assert np.isfinite(current).all()
        assert np.isnan(future).all()

        current, future, _ = self._read(
            PeriodCurrentFuture(3), band, window, n_blocks=1
        )
        assert np.isnan(current).all()
        assert (future == 0).all()

    @pytest.mark.parametrize("n_blocks", [1, 2])
    def test_band_appends_its_tail_in_the_reported_order(self, windows, n_blocks):
        """A band joined over blocks, extended by its tail, as callers see it."""
        window = windows.windows[1]
        band = _increment([1.0, 2.0], window.eval_dates, draws=4)

        result = self._read(PeriodBand(), band, window, n_blocks=n_blocks)

        np.testing.assert_array_equal(result["geo"], ["g1", "g0"])
        np.testing.assert_array_equal(
            result.coords["observed"], [True, True, False, False]
        )
        xr.testing.assert_identical(
            result.sel(realization_date=window.eval_dates).drop_vars("observed"),
            band.sel(self.CONTEXT.reported_order),
        )

    def test_total_is_labelled_by_the_period_end(self, windows):
        """A total gets a length-one ``date``, the period's end."""
        window = windows.windows[0]
        total = _increment([1.0, 2.0], window.eval_dates[:2], draws=2).sum(
            "realization_date"
        )

        result = self._read(PeriodTotal(), total, window, n_blocks=2)

        assert result.sizes["date"] == 1
        assert pd.Timestamp(result["date"].values[0]) == window.end
        xr.testing.assert_identical(
            result.squeeze("date", drop=True),
            total.sel(self.CONTEXT.reported_order),
        )

    def test_reduce_row(self):
        """A total sums the per-date increment; a band and a split keep it per date."""
        reducer = IdentityLinkReducer(scale=2.0)
        delta_mu = xr.DataArray(
            np.arange(6.0).reshape(2, 3),
            dims=("sample", "date"),
            coords={"date": DATES[:3]},
        )

        xr.testing.assert_identical(
            PeriodTotal().reduce_row(delta_mu, reducer),
            reducer.counterfactual_minus_baseline(delta_mu),
        )
        per_date = reducer.per_date_increment(delta_mu).rename(date="realization_date")
        for readout in (PeriodBand(), PeriodCurrentFuture(0)):
            xr.testing.assert_identical(readout.reduce_row(delta_mu, reducer), per_date)

    @pytest.mark.parametrize("link", ["identity", "log"])
    def test_reduce_rows_matches_reducing_each_row(self, link):
        """Stacking the rows before the reduction gives bitwise the same increment.

        A band and a split reduce the stacked rows in one call; a total sums
        each row over ``date`` first.  Both have to equal reducing every row on
        its own and stacking the results.
        """
        rng = np.random.default_rng(0)
        channels = ["c0", "c1", "c2"]
        rows = [
            xr.DataArray(
                rng.normal(size=(4, 3, 2)),
                dims=("sample", "date", "geo"),
                coords={"date": DATES[:3], "geo": ["g0", "g1"]},
            )
            for _ in channels
        ]
        if link == "identity":
            reducer = IdentityLinkReducer(
                scale=xr.DataArray([2.0, 3.0], dims="geo", coords={"geo": ["g0", "g1"]})
            )
        else:
            reducer = LogLinkReducer(
                baseline_response=xr.DataArray(
                    rng.uniform(1, 2, size=(4, 5, 2)),
                    dims=("sample", "date", "geo"),
                    coords={"date": DATES[:5], "geo": ["g0", "g1"]},
                )
            )

        for readout in (PeriodTotal(), PeriodBand(), PeriodCurrentFuture(0)):
            expected = xr.concat(
                [readout.reduce_row(row, reducer) for row in rows], dim="channel"
            ).assign_coords(channel=channels)
            got = readout.reduce_rows(iter(rows), reducer, channels)
            # The caller transposes to its own order, so only the values and
            # labels have to match.
            xr.testing.assert_identical(got.transpose(*expected.dims), expected)
            xr.testing.assert_identical(
                readout.reduce_rows(iter(rows[:1]), reducer, None),
                readout.reduce_row(rows[0], reducer),
            )
