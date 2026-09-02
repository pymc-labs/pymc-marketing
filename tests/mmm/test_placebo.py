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
"""Tests for the placebo (negative-control) channel utilities."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.mmm.placebo import (
    add_placebo_channel,
    contribution_share,
    placebo_correlation,
    summarize_placebo_contribution,
)


@pytest.fixture
def spend() -> pd.DataFrame:
    """Two geos of autocorrelated spend, so a shift has something to preserve."""
    rng = np.random.default_rng(20260902)
    rows = []
    for geo in ("geo_a", "geo_b"):
        x = np.zeros(200)
        for i in range(1, 200):
            x[i] = 0.85 * x[i - 1] + rng.normal(scale=1.0)
        rows.append(pd.DataFrame({"geo": geo, "x1": x - x.min() + 1.0}))
    return pd.concat(rows, ignore_index=True)


def _contribution(values: dict[str, float], seed: int = 0) -> xr.DataArray:
    """A posterior of channel contributions with the given per-channel means."""
    rng = np.random.default_rng(seed)
    channels = list(values)
    data = np.stack(
        [rng.normal(loc=values[c], scale=0.01, size=(2, 250, 5)) for c in channels],
        axis=-1,
    )
    return xr.DataArray(
        data,
        dims=("chain", "draw", "date", "channel"),
        coords={"channel": channels},
    )


class TestAddPlaceboChannel:
    def test_shift_preserves_the_marginal_distribution(self, spend):
        out, name = add_placebo_channel(spend, "x1", group_column="geo", seed=1)
        for _, group in out.groupby("geo", sort=False):
            np.testing.assert_allclose(
                np.sort(group["x1"].to_numpy()), np.sort(group[name].to_numpy())
            )

    def test_shift_preserves_autocorrelation(self, spend):
        out, name = add_placebo_channel(spend, "x1", group_column="geo", seed=1)
        g = out[out.geo == "geo_a"]
        real, placebo = g["x1"].to_numpy(), g[name].to_numpy()
        ac = lambda v: np.corrcoef(v[:-1], v[1:])[0, 1]  # noqa: E731
        assert ac(placebo) == pytest.approx(ac(real), abs=0.05)

    def test_permute_destroys_autocorrelation(self, spend):
        out, name = add_placebo_channel(
            spend, "x1", group_column="geo", method="permute", seed=1
        )
        g = out[out.geo == "geo_a"]
        v = g[name].to_numpy()
        assert abs(np.corrcoef(v[:-1], v[1:])[0, 1]) < 0.3

    def test_alignment_is_destroyed(self, spend):
        out, name = add_placebo_channel(spend, "x1", group_column="geo", seed=1)
        assert abs(placebo_correlation(out, "x1", name)) <= 0.3

    def test_seed_is_deterministic(self, spend):
        a, name = add_placebo_channel(spend, "x1", group_column="geo", seed=7)
        b, _ = add_placebo_channel(spend, "x1", group_column="geo", seed=7)
        pd.testing.assert_series_equal(a[name], b[name])

    @pytest.mark.parametrize("seed", range(25))
    def test_the_guard_holds_for_every_seed(self, spend, seed):
        """A near-copy is not a placebo, and an unseeded caller must not get one."""
        out, name = add_placebo_channel(spend, "x1", group_column="geo", seed=seed)
        assert abs(placebo_correlation(out, "x1", name)) <= 0.3

    def test_a_loose_bound_admits_what_the_default_rejects(self, spend):
        """Positive control: the guard is doing the work, not the shift itself.

        With the bound effectively switched off some seed produces a placebo the
        default would refuse. If this never happened, the guard would be
        untested and passing for the wrong reason.
        """
        worst = 0.0
        for seed in range(60):
            out, name = add_placebo_channel(
                spend, "x1", group_column="geo", seed=seed, max_abs_corr=1.0
            )
            worst = max(worst, abs(placebo_correlation(out, "x1", name)))
        assert worst > 0.3

    def test_unreachable_bound_raises(self, spend):
        with pytest.raises(ValueError, match="no draw left"):
            add_placebo_channel(
                spend, "x1", group_column="geo", seed=1, max_abs_corr=0.0, max_tries=5
            )

    def test_unknown_method_raises(self, spend):
        with pytest.raises(ValueError, match="unknown method"):
            add_placebo_channel(spend, "x1", method="nope")

    def test_the_real_column_is_untouched(self, spend):
        out, _ = add_placebo_channel(spend, "x1", group_column="geo", seed=1)
        pd.testing.assert_series_equal(out["x1"], spend["x1"])


class TestSummarize:
    @pytest.fixture
    def fits(self):
        with_placebo = _contribution({"x1": 0.45, "x2": 0.30, "x1_placebo": 0.25})
        without_placebo = _contribution({"x1": 0.60, "x2": 0.40}, seed=1)
        return with_placebo, without_placebo

    def test_shares_sum_to_one(self, fits):
        share = contribution_share(fits[0]).mean(dim=("chain", "draw"))
        assert float(share.sum()) == pytest.approx(1.0)

    def test_columns_and_ordering(self, fits):
        out = summarize_placebo_contribution(*fits, "x1_placebo")
        assert list(out.columns) == [
            "share",
            "hdi_low",
            "hdi_high",
            "prob_exceeds_placebo",
            "share_without_placebo",
            "lost",
            "dilution_alone",
        ]
        assert out["share"].is_monotonic_decreasing

    def test_placebo_row_has_no_comparison_values(self, fits):
        out = summarize_placebo_contribution(*fits, "x1_placebo")
        row = out.loc["x1_placebo"]
        for col in (
            "prob_exceeds_placebo",
            "share_without_placebo",
            "lost",
            "dilution_alone",
        ):
            assert np.isnan(row[col])

    def test_lost_and_dilution_are_the_stated_arithmetic(self, fits):
        out = summarize_placebo_contribution(*fits, "x1_placebo")
        floor = out.loc["x1_placebo", "share"]
        for channel in ("x1", "x2"):
            row = out.loc[channel]
            assert row["lost"] == pytest.approx(
                row["share_without_placebo"] - row["share"]
            )
            assert row["dilution_alone"] == pytest.approx(
                row["share_without_placebo"] * floor
            )

    def test_hdi_prob_is_actually_applied(self, fits):
        """A wider mass must give a wider interval.

        Checking only the returned columns would pass even if ``hdi_prob`` were
        silently swallowed and every caller left on the default.
        """
        narrow = summarize_placebo_contribution(*fits, "x1_placebo", hdi_prob=0.50)
        wide = summarize_placebo_contribution(*fits, "x1_placebo", hdi_prob=0.99)
        for channel in narrow.index:
            n = narrow.loc[channel, "hdi_high"] - narrow.loc[channel, "hdi_low"]
            w = wide.loc[channel, "hdi_high"] - wide.loc[channel, "hdi_low"]
            assert w > n

    def test_missing_placebo_raises(self, fits):
        with pytest.raises(ValueError, match="not a channel"):
            summarize_placebo_contribution(*fits, "not_a_channel")

    def test_placebo_in_the_baseline_raises(self, fits):
        with_placebo, _ = fits
        with pytest.raises(ValueError, match="appears in the baseline fit"):
            summarize_placebo_contribution(with_placebo, with_placebo, "x1_placebo")
