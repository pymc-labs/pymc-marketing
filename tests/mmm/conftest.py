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
"""Shared fixtures for MMM plotting tests."""

import arviz as az
import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def mock_posterior_data():
    """Mock posterior Dataset for testing data parameters."""
    rng = np.random.default_rng(42)
    return xr.Dataset(
        {
            "intercept": xr.DataArray(
                rng.normal(size=(4, 100, 52)),
                dims=("chain", "draw", "date"),
                coords={
                    "chain": np.arange(4),
                    "draw": np.arange(100),
                    "date": pd.date_range("2025-01-01", periods=52, freq="W"),
                },
            )
        }
    )


@pytest.fixture
def mock_constant_data():
    """Mock constant_data Dataset for saturation plots."""
    rng = np.random.default_rng(42)
    n_dates = 52
    n_channels = 3

    return xr.Dataset(
        {
            "channel_data": xr.DataArray(
                rng.uniform(0, 100, size=(n_dates, n_channels)),
                dims=("date", "channel"),
                coords={
                    "date": pd.date_range("2025-01-01", periods=n_dates, freq="W"),
                    "channel": ["TV", "Radio", "Digital"],
                },
            ),
            "channel_scale": xr.DataArray(
                rng.uniform(0.5, 2.0, size=(n_channels,)),
                dims=("channel",),
                coords={"channel": ["TV", "Radio", "Digital"]},
            ),
            "target_scale": xr.DataArray(1.0),
        }
    )


@pytest.fixture
def mock_sensitivity_data():
    """Mock sensitivity analysis data."""
    rng = np.random.default_rng(42)
    return xr.Dataset(
        {
            "x": xr.DataArray(
                rng.normal(size=(100, 20)),
                dims=("sample", "sweep"),
                coords={
                    "sample": np.arange(100),
                    "sweep": np.linspace(0, 1, 20),
                },
            )
        }
    )


@pytest.fixture
def mock_idata_with_posterior():
    """Mock InferenceData with posterior data."""
    rng = np.random.default_rng(42)
    posterior = xr.Dataset(
        {
            "intercept": xr.DataArray(
                rng.normal(size=(4, 100, 52)),
                dims=("chain", "draw", "date"),
                coords={
                    "chain": np.arange(4),
                    "draw": np.arange(100),
                    "date": pd.date_range("2025-01-01", periods=52, freq="W"),
                },
            )
        }
    )
    return az.InferenceData(posterior=posterior)


@pytest.fixture
def mock_idata_with_uplift_curve():
    """Mock InferenceData with uplift_curve in sensitivity_analysis."""
    rng = np.random.default_rng(42)

    posterior = xr.Dataset(
        {
            "intercept": xr.DataArray(
                rng.normal(size=(4, 100)),
                dims=("chain", "draw"),
            )
        }
    )

    sensitivity_analysis = xr.Dataset(
        {
            "uplift_curve": xr.DataArray(
                rng.normal(size=(100, 20)),
                dims=("sample", "sweep"),
            )
        }
    )

    return az.InferenceData(
        posterior=posterior, sensitivity_analysis=sensitivity_analysis
    )


@pytest.fixture
def mock_idata_with_sensitivity():
    """Mock InferenceData with sensitivity_analysis group."""
    rng = np.random.default_rng(42)

    posterior = xr.Dataset(
        {
            "intercept": xr.DataArray(
                rng.normal(size=(4, 100)),
                dims=("chain", "draw"),
            )
        }
    )

    sensitivity_analysis = xr.Dataset(
        {
            "x": xr.DataArray(
                rng.normal(size=(100, 20)),
                dims=("sample", "sweep"),
                coords={
                    "sample": np.arange(100),
                    "sweep": np.linspace(0, 1, 20),
                },
            )
        }
    )

    return az.InferenceData(
        posterior=posterior, sensitivity_analysis=sensitivity_analysis
    )


@pytest.fixture
def mock_idata_for_legacy():
    """Mock InferenceData for legacy suite tests."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=52, freq="W")

    posterior_predictive = xr.Dataset(
        {
            "y": xr.DataArray(
                rng.normal(size=(4, 100, 52)),
                dims=("chain", "draw", "date"),
                coords={
                    "chain": np.arange(4),
                    "draw": np.arange(100),
                    "date": dates,
                },
            )
        }
    )

    return az.InferenceData(posterior_predictive=posterior_predictive)
