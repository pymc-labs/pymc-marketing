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

from pymc_marketing.mmm.placebo import (
    add_placebo_channel,
    summarize_placebo_contribution,
)


@pytest.fixture
def spend_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 104
    base = np.abs(np.sin(np.arange(n) / 6.0)) * 100
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="W"),
            "TV": base + rng.normal(0, 5, n),
            "Radio": rng.uniform(0, 50, n),
        }
    )


def test_placebo_preserves_marginal_distribution(spend_data) -> None:
    X_out, name = add_placebo_channel(spend_data, "TV", method="permute", seed=1)
    assert name == "TV_placebo"
    np.testing.assert_allclose(
        np.sort(X_out[name].to_numpy()), np.sort(spend_data["TV"].to_numpy())
    )
    # original frame untouched
    assert "TV_placebo" not in spend_data.columns


def test_placebo_destroys_time_alignment(spend_data) -> None:
    X_out, name = add_placebo_channel(spend_data, "TV", method="permute", seed=1)
    original = spend_data["TV"].to_numpy()
    corr = np.corrcoef(original, X_out[name].to_numpy())[0, 1]
    assert abs(corr) < 0.3
    assert not np.array_equal(original, X_out[name].to_numpy())


def test_shift_preserves_autocorrelation_structure(spend_data) -> None:
    X_out, name = add_placebo_channel(spend_data, "TV", method="shift", seed=3)

    def lag1(x):
        return np.corrcoef(x[:-1], x[1:])[0, 1]

    original = spend_data["TV"].to_numpy()
    shifted = X_out[name].to_numpy()
    # circular shift preserves serial structure far better than permutation
    assert lag1(shifted) > 0.8 * lag1(original)
    assert not np.array_equal(original, shifted)


def test_seed_determinism_and_errors(spend_data) -> None:
    a, _ = add_placebo_channel(spend_data, "TV", seed=7)
    b, _ = add_placebo_channel(spend_data, "TV", seed=7)
    pd.testing.assert_frame_equal(a, b)

    with pytest.raises(KeyError):
        add_placebo_channel(spend_data, "NotAChannel")
    with pytest.raises(ValueError, match="already exists"):
        add_placebo_channel(a, "TV")
    with pytest.raises(ValueError, match="unknown method"):
        add_placebo_channel(spend_data, "TV", method="bogus")


@pytest.fixture
def contribution() -> xr.DataArray:
    rng = np.random.default_rng(0)
    chains, draws, dates = 2, 200, 30
    real = np.abs(rng.normal(100, 10, size=(chains, draws, dates)))
    weak = np.abs(rng.normal(20, 8, size=(chains, draws, dates)))
    placebo = np.abs(rng.normal(5, 4, size=(chains, draws, dates)))
    return xr.DataArray(
        np.stack([real, weak, placebo], axis=-1),
        dims=("chain", "draw", "date", "channel"),
        coords={"channel": ["TV", "Radio", "TV_placebo"]},
    )


def test_summary_ranks_channels_against_placebo_floor(contribution) -> None:
    summary = summarize_placebo_contribution(contribution, "TV_placebo")
    assert list(summary.columns) == [
        "channel",
        "share_mean",
        "share_hdi_low",
        "share_hdi_high",
        "prob_exceeds_placebo",
        "is_placebo",
    ]
    by_channel = summary.set_index("channel")
    # a strong real channel clearly exceeds the placebo floor
    assert by_channel.loc["TV", "prob_exceeds_placebo"] > 0.99
    # HDI brackets the mean, shares sum to ~1
    for _, row in summary.iterrows():
        assert row["share_hdi_low"] <= row["share_mean"] <= row["share_hdi_high"]
    assert summary["share_mean"].sum() == pytest.approx(1.0, abs=1e-6)
    assert by_channel.loc["TV_placebo", "is_placebo"]
    assert np.isnan(by_channel.loc["TV_placebo", "prob_exceeds_placebo"])


def test_summary_unknown_placebo_raises(contribution) -> None:
    with pytest.raises(KeyError):
        summarize_placebo_contribution(contribution, "nope")
