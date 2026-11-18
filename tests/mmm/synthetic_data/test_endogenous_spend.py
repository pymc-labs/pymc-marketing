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
"""Tests for the endogenous-spend synthetic market generator."""

import numpy as np
import pandas as pd
import pytest

from pymc_marketing.mmm.budget_model import lift_test_design
from pymc_marketing.mmm.synthetic_data import (
    EndogenousSpendMarket,
    simulate_endogenous_spend_market,
    tv_marginal_return,
)
from pymc_marketing.mmm.synthetic_data.endogenous_spend import TV_BETA, TV_LAM

SCENARIOS = ["forecast", "target_chasing", "search", "observed_only"]


@pytest.fixture(scope="module", params=SCENARIOS)
def market(request) -> EndogenousSpendMarket:
    return simulate_endogenous_spend_market(request.param, random_seed=1)


def test_schema(market):
    assert isinstance(market, EndogenousSpendMarket)
    assert list(market.data.columns) == [
        "date",
        "tv",
        "digital",
        "inflation",
        "unemployment",
        "y",
    ]
    assert len(market.data) == 104 + 4 + 12
    assert market.n_history == 104
    assert market.demand_shock.shape == (len(market.data),)
    assert pd.infer_freq(market.data["date"]) == "W-MON"
    assert not market.data.isna().any().any()


def test_controls_are_standardised(market):
    for column in ["inflation", "unemployment"]:
        assert np.isclose(market.data[column].mean(), 0.0, atol=1e-12)
        assert np.isclose(market.data[column].std(ddof=0), 1.0)


def test_spend_floor(market):
    assert (market.data["digital"] >= 0.1).all()
    history = market.data["tv"].iloc[: market.n_history]
    assert (history >= 0.1).all()


def test_reproducible_with_seed(market):
    again = simulate_endogenous_spend_market(market.scenario, random_seed=1)
    pd.testing.assert_frame_equal(market.data, again.data)
    pd.testing.assert_frame_equal(market.lift_test, again.lift_test)
    np.testing.assert_array_equal(market.demand_shock, again.demand_shock)


def test_different_seeds_differ(market):
    other = simulate_endogenous_spend_market(market.scenario, random_seed=2)
    assert not np.allclose(market.data["y"], other.data["y"])


def test_accepts_generator(market):
    rng = np.random.default_rng(1)
    from_rng = simulate_endogenous_spend_market(market.scenario, random_seed=rng)
    pd.testing.assert_frame_equal(market.data, from_rng.data)


def test_design_matches_experiment_window(market):
    design = market.design
    dates = market.data["date"]
    assert design["channel"].item() == "tv"
    assert design["start_date"].item() == dates.iloc[market.n_history]
    assert design["end_date"].item() == dates.iloc[market.n_history + 3]
    shift, holdout = lift_test_design(design, dates=dates, channels=["tv", "digital"])
    if market.scenario == "search":
        assert design["mode"].item() == "set"
        assert np.isnan(design["delta_x"].item())
        assert int(holdout.sum()) == 4
        assert float(np.abs(shift).sum()) == 0.0
    else:
        assert design["mode"].item() == "shift"
        assert design["delta_x"].item() == -1.5
        assert int(holdout.sum()) == 0
        assert np.isclose(float(shift.sum()), -6.0)


def test_lift_test_format(market):
    lift = market.lift_test
    assert list(lift.columns) == ["channel", "x", "delta_x", "delta_y", "sigma"]
    assert lift["channel"].item() == "tv"
    assert lift["sigma"].item() == 1.0
    assert lift["x"].item() > 0
    assert lift["delta_x"].item() < 0
    assert lift["delta_y"].item() < 0


def test_answer_key_is_consistent(market):
    assert market.operating_point > 0
    assert market.true_marginal_return == tv_marginal_return(market.operating_point)


def test_search_scenario_caps_and_zeroes_spend():
    market = simulate_endogenous_spend_market("search", random_seed=1, search_cap=6.0)
    tv = market.data["tv"].to_numpy()
    in_test = slice(market.n_history, market.n_history + 4)
    assert (tv[in_test] == 0.0).all()
    assert tv.max() <= 6.0
    assert np.isclose(market.lift_test["delta_x"].item(), -market.lift_test["x"].item())


def test_shift_lift_matches_design_without_feedback():
    market = simulate_endogenous_spend_market(
        "forecast", random_seed=1, test_delta=-1.0
    )
    assert np.isclose(market.lift_test["delta_x"].item(), -1.0)


@pytest.mark.parametrize(
    "scenario, sign",
    [("forecast", 1), ("target_chasing", -1)],
    ids=["procyclical", "countercyclical"],
)
def test_spend_tracks_unrecorded_demand(scenario, sign):
    market = simulate_endogenous_spend_market(scenario, random_seed=1)
    history = slice(0, market.n_history)
    corr = np.corrcoef(market.data["tv"].iloc[history], market.demand_shock[history])[
        0, 1
    ]
    assert sign * corr > 0.1


def test_observed_only_spend_ignores_unrecorded_demand():
    market = simulate_endogenous_spend_market("observed_only", random_seed=1)
    history = slice(0, market.n_history)
    corr = np.corrcoef(market.data["tv"].iloc[history], market.demand_shock[history])[
        0, 1
    ]
    assert abs(corr) < 0.1


def test_custom_lengths():
    market = simulate_endogenous_spend_market(
        "forecast", random_seed=1, n_history=20, n_test=2, n_followup=3
    )
    assert len(market.data) == 25
    assert market.design["start_date"].item() == market.data["date"].iloc[20]
    assert market.design["end_date"].item() == market.data["date"].iloc[21]


def test_lift_sigma_is_reported():
    market = simulate_endogenous_spend_market("forecast", random_seed=1, lift_sigma=2.5)
    assert market.lift_test["sigma"].item() == 2.5


def test_invalid_scenario():
    with pytest.raises(ValueError, match="scenario must be one of"):
        simulate_endogenous_spend_market("random")


@pytest.mark.parametrize("spend", [0.5, 5.0, 20.0])
def test_tv_marginal_return_is_derivative_of_saturation(spend):
    def contribution(x: float) -> float:
        return TV_BETA * x / (TV_LAM + x)

    eps = 1e-6
    numeric = (contribution(spend + eps) - contribution(spend - eps)) / (2 * eps)
    assert np.isclose(tv_marginal_return(spend), numeric, rtol=1e-6)
