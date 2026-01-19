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
"""Shared fixtures for MMM tests."""

import numpy as np
import pandas as pd
import pytest

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.multidimensional import MMM

# ============================================================================
# Data Fixtures
# ============================================================================


@pytest.fixture
def simple_mmm_data():
    """Create simple single-dimension MMM data.

    Returns dict with:
    - X: DataFrame with date and 3 channels (14 periods)
    - y: Series with target values
    """
    date_range = pd.date_range("2023-01-01", periods=14)
    np.random.seed(42)

    channel_1 = np.random.randint(100, 500, size=len(date_range))
    channel_2 = np.random.randint(100, 500, size=len(date_range))
    channel_3 = np.random.randint(100, 500, size=len(date_range))

    X = pd.DataFrame(
        {
            "date": date_range,
            "channel_1": channel_1,
            "channel_2": channel_2,
            "channel_3": channel_3,
        }
    )
    y = pd.Series(
        channel_1
        + channel_2
        + channel_3
        + np.random.randint(100, 300, size=len(date_range)),
        name="target",
    )

    return {"X": X, "y": y}


@pytest.fixture
def panel_mmm_data():
    """Create panel (multidimensional) MMM data with country dimension.

    Returns dict with:
    - X: DataFrame with date, country, and 2 channels (7 periods × 2 countries)
    - y: Series with target values
    """
    date_range = pd.date_range("2023-01-01", periods=7)
    countries = ["US", "UK"]
    np.random.seed(123)

    records = []
    for country in countries:
        for date in date_range:
            channel_1 = np.random.randint(100, 500)
            channel_2 = np.random.randint(100, 500)
            target = channel_1 + channel_2 + np.random.randint(50, 150)
            records.append((date, country, channel_1, channel_2, target))

    df = pd.DataFrame(
        records,
        columns=["date", "country", "channel_1", "channel_2", "target"],
    )

    X = df[["date", "country", "channel_1", "channel_2"]].copy()
    y = df["target"].copy()

    return {"X": X, "y": y}


# ============================================================================
# Fitted Model Fixtures
# ============================================================================


@pytest.fixture
def simple_fitted_mmm(simple_mmm_data, mock_pymc_sample):
    """Create a simple fitted MMM for testing (no extra dimensions)."""
    X = simple_mmm_data["X"]
    y = simple_mmm_data["y"]

    mmm = MMM(
        channel_columns=["channel_1", "channel_2", "channel_3"],
        date_column="date",
        target_column="target",
        control_columns=None,
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )

    mmm.fit(X, y)

    return mmm


@pytest.fixture
def panel_fitted_mmm(panel_mmm_data, mock_pymc_sample):
    """Create a panel (multidimensional) fitted MMM for testing."""
    X = panel_mmm_data["X"]
    y = panel_mmm_data["y"]

    mmm = MMM(
        channel_columns=["channel_1", "channel_2"],
        date_column="date",
        target_column="target",
        dims=("country",),
        control_columns=None,
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )

    mmm.fit(X, y)

    return mmm
