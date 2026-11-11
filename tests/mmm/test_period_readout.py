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
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.mmm.counterfactual import EvaluationWindows
from pymc_marketing.mmm.period_readout import LinearPredictorDelta

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
