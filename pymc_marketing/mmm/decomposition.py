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
"""Shared decomposition math helpers for MMM counterfactuals."""

from __future__ import annotations

import numpy as np
import xarray as xr


def original_scale_prediction_from_mu(
    mu: xr.DataArray,
    target_scale: xr.DataArray,
) -> xr.DataArray:
    """Convert log-link linear predictor samples to original scale."""
    return np.exp(mu) * target_scale


def log_counterfactual_remove_component(
    mu_total: xr.DataArray,
    component: xr.DataArray,
    target_scale: xr.DataArray,
) -> xr.DataArray:
    """Per-draw log-link counterfactual contribution for a component."""
    y_hat = original_scale_prediction_from_mu(mu_total, target_scale)
    y_hat_without_component = np.exp(mu_total - component) * target_scale
    return y_hat - y_hat_without_component


def identity_counterfactual_component(
    component: xr.DataArray,
    target_scale: xr.DataArray,
) -> xr.DataArray:
    """Per-draw identity-link contribution for a component."""
    return component * target_scale


def safe_proportional_share(
    numerator: xr.DataArray,
    denominator: xr.DataArray,
) -> xr.DataArray:
    """Compute a finite proportional share, guarding against zero denominators."""
    denom_safe = xr.where(np.abs(denominator) > 1e-12, denominator, np.nan)
    share = numerator / denom_safe
    return share.fillna(0.0).where(np.isfinite(share), 0.0)
