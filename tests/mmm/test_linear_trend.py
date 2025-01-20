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
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytest

from pymc_marketing.mmm.linear_trend import LinearTrend
from pymc_marketing.prior import Prior


def test_init_errors_with_additional_parameter() -> None:
    priors = {
        "delta": Prior("Normal"),
        "k": Prior("Normal"),
    }

    match = "Invalid priors"
    with pytest.raises(ValueError, match=match):
        LinearTrend(priors=priors, include_intercept=False)


def test_init_errors_with_non_dim_subset() -> None:
    priors = {
        "delta": Prior("Normal", dims=("changepoint", "geo")),
    }

    match = "Invalid dimensions"
    with pytest.raises(ValueError, match=match):
        LinearTrend(priors=priors)


@pytest.mark.parametrize(
    "include_intercept, expected_keys",
    [(True, {"delta", "k"}), (False, {"delta"})],
    ids=["with_intercept", "without_intercept"],
)
def test_defaults_priors(include_intercept, expected_keys) -> None:
    trend = LinearTrend(include_intercept=include_intercept)

    assert set(trend.priors.keys()) == expected_keys


def test_default_prior_includes_changepoint() -> None:
    trend = LinearTrend()
    assert trend.priors["delta"].dims == ("changepoint",)


@pytest.mark.parametrize(
    "include_intercept, expected_keys",
    [(True, {"delta", "k"}), (False, {"delta"})],
    ids=["with_intercept", "without_intercept"],
)
def test_apply(include_intercept, expected_keys) -> None:
    trend = LinearTrend(include_intercept=include_intercept)

    n_obs = 100
    x = np.linspace(0, 1, n_obs)
    with pm.Model() as model:
        mu = trend.apply(x)

    assert mu.eval().shape == (n_obs,)
    assert set(model.named_vars.keys()) == expected_keys


@pytest.mark.parametrize(
    "delta_dims",
    [None, ("changepoint",), ("changepoint", "geo")],
    ids=["scalar", "1d", "2d"],
)
def test_apply_additional_dims(delta_dims) -> None:
    priors = {
        "delta": Prior("Normal", dims=delta_dims),
    }
    trend = LinearTrend(priors=priors, dims=("geo",))

    n_obs = 100
    x = np.linspace(0, 1, n_obs)
    geos = ["A", "B", "C"]
    coords = {
        "geo": geos,
    }
    with pm.Model(coords=coords):
        mu = trend.apply(x)

    n_cols = len(geos) if delta_dims is not None and "geo" in delta_dims else 1
    assert mu.eval().shape == (n_obs, n_cols)


@pytest.mark.parametrize(
    "include_changepoints",
    [True, False],
    ids=["with_changepoint", "without_changepoint"],
)
def test_plot_workflow(include_changepoints: bool) -> None:
    trend = LinearTrend()

    prior = trend.sample_prior()
    curve = trend.sample_curve(prior)
    fig, axes = trend.plot_curve(curve, include_changepoints=include_changepoints)

    assert isinstance(fig, plt.Figure)
    assert axes.size == 1
    assert isinstance(axes[0], plt.Axes)
