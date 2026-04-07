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
