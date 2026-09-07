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
"""Atomic tests for decomposition math helpers."""

import numpy as np
import xarray as xr

from pymc_marketing.mmm.decomposition import (
    identity_counterfactual_component,
    log_counterfactual_remove_component,
    original_scale_prediction_from_mu,
    safe_proportional_share,
)


def test_original_scale_prediction_from_mu_matches_exp_mu_times_scale():
    """Prediction helper computes exp(mu) * target_scale with xarray broadcasting."""
    mu = xr.DataArray(
        [[[0.0, 1.0]]],
        dims=("chain", "draw", "date"),
        coords={"date": ["2024-01-01", "2024-01-08"]},
    )
    target_scale = xr.DataArray(100.0)

    result = original_scale_prediction_from_mu(mu=mu, target_scale=target_scale)
    expected = np.exp(mu) * target_scale
    xr.testing.assert_allclose(result, expected)


def test_log_counterfactual_remove_component_matches_closed_form():
    """Log counterfactual helper matches [exp(mu)-exp(mu-v)] * scale exactly."""
    mu = xr.DataArray(
        [[[[2.0], [2.2]]]],
        dims=("chain", "draw", "date", "country"),
        coords={"date": ["2024-01-01", "2024-01-08"], "country": ["US"]},
    )
    component = xr.DataArray(
        [[[[0.3], [0.1]]]],
        dims=("chain", "draw", "date", "country"),
        coords={"date": ["2024-01-01", "2024-01-08"], "country": ["US"]},
    )
    target_scale = xr.DataArray(50.0)

    result = log_counterfactual_remove_component(
        mu_total=mu,
        component=component,
        target_scale=target_scale,
    )
    expected = (np.exp(mu) - np.exp(mu - component)) * target_scale
    xr.testing.assert_allclose(result, expected)


def test_identity_counterfactual_component_is_component_times_scale():
    """Identity helper is a direct multiplication by target_scale."""
    component = xr.DataArray(
        [[[[1.0, 2.0]]]],
        dims=("chain", "draw", "date", "channel"),
        coords={"date": ["2024-01-01"], "channel": ["TV", "Radio"]},
    )
    target_scale = xr.DataArray(10.0)

    result = identity_counterfactual_component(
        component=component,
        target_scale=target_scale,
    )
    expected = component * target_scale
    xr.testing.assert_allclose(result, expected)


def test_safe_proportional_share_returns_zero_when_denominator_is_zero():
    """Zero denominator produces zero share values (no NaN/inf)."""
    numerator = xr.DataArray([1.0, -2.0, 0.0], dims=("date",))
    denominator = xr.DataArray([0.0, 0.0, 0.0], dims=("date",))

    result = safe_proportional_share(numerator=numerator, denominator=denominator)
    expected = xr.DataArray([0.0, 0.0, 0.0], dims=("date",))
    xr.testing.assert_allclose(result, expected)


def test_safe_proportional_share_removes_inf_and_nan():
    """Near-zero and zero denominators never leak NaN/inf to outputs."""
    numerator = xr.DataArray([1.0, 2.0, -3.0], dims=("date",))
    denominator = xr.DataArray([0.0, 1e-20, -1e-20], dims=("date",))

    result = safe_proportional_share(numerator=numerator, denominator=denominator)
    assert np.isfinite(result.values).all()
    xr.testing.assert_allclose(result, xr.zeros_like(result))


def test_safe_proportional_share_matches_normal_division_when_denominator_nonzero():
    """With safe nonzero denominator, helper equals plain division."""
    numerator = xr.DataArray([2.0, 6.0, -4.0], dims=("date",))
    denominator = xr.DataArray([1.0, 2.0, -2.0], dims=("date",))

    result = safe_proportional_share(numerator=numerator, denominator=denominator)
    expected = numerator / denominator
    xr.testing.assert_allclose(result, expected)


def _toy_log_posterior() -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Small (chain, draw) log-link posterior for difference-of-means checks."""
    rng = np.random.default_rng(0)
    mu = xr.DataArray(
        rng.normal(0.0, 0.5, size=(2, 50, 3)),
        dims=("chain", "draw", "date"),
    )
    component = xr.DataArray(
        rng.normal(0.0, 0.2, size=(2, 50, 3)),
        dims=("chain", "draw", "date"),
    )
    target_scale = xr.DataArray(100.0)
    return mu, component, target_scale


def test_mean_is_difference_of_means_for_finite_samples():
    r"""E[exp(mu)*s - exp(mu-v)*s] equals E[exp(mu)]*s - E[exp(mu-v)]*s exactly.

    Locks in the linearity-of-expectation identity invoked in the docstrings:
    the implemented per-draw form and the difference-of-means form share the
    same posterior mean.
    """
    mu, component, target_scale = _toy_log_posterior()

    per_draw = log_counterfactual_remove_component(
        mu_total=mu, component=component, target_scale=target_scale
    )
    mean_of_difference = per_draw.mean(("chain", "draw"))

    y_hat_mean = (np.exp(mu) * target_scale).mean(("chain", "draw"))
    y_hat_no_comp_mean = (np.exp(mu - component) * target_scale).mean(("chain", "draw"))
    difference_of_means = y_hat_mean - y_hat_no_comp_mean

    xr.testing.assert_allclose(mean_of_difference, difference_of_means)


def test_per_draw_form_differs_from_difference_of_means_in_spread():
    """The per-draw form carries draw-level variation the mean-difference loses.

    Credible intervals are only well defined for the per-draw counterfactual,
    so its variance across draws must be strictly positive (whereas a
    difference of two posterior means is a single number with no spread).
    """
    mu, component, target_scale = _toy_log_posterior()

    per_draw = log_counterfactual_remove_component(
        mu_total=mu, component=component, target_scale=target_scale
    )
    spread = per_draw.std(("chain", "draw"))
    assert (spread.values > 0).all()
