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

import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytest
from pymc_extras.prior import Prior

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation, LogSaturation
from pymc_marketing.mmm.components.adstock import WeibullCDFAdstock
from pymc_marketing.mmm.mmm import MMM
from pymc_marketing.special_priors import LogNormalPrior

seed: int = sum(map(ord, "pymc_marketing"))
rng: np.random.Generator = np.random.default_rng(seed=seed)


# ============================================================================
# Data Fixtures
# ============================================================================


def _make_mmm_data(
    start_date: str = "2023-01-01",
    periods: int = 14,
    freq: str = "W",
    n_channels: int = 3,
    seed: int = 42,
) -> dict:
    """Build synthetic MMM data with the given parameters.

    Parameters
    ----------
    start_date : str
        Start date for the date range.
    periods : int
        Number of time periods.
    freq : str
        Pandas frequency string (e.g. ``"W"``, ``"MS"``).
    n_channels : int
        Number of media channels to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{"X": pd.DataFrame, "y": pd.Series}``
    """
    date_range = pd.date_range(start_date, periods=periods, freq=freq)
    np.random.seed(seed)

    channels = {
        f"channel_{i + 1}": np.random.uniform(100, 500, size=len(date_range))
        for i in range(n_channels)
    }

    X = pd.DataFrame({"date": date_range, **channels})
    y = pd.Series(
        sum(channels.values()) + np.random.randint(100, 300, size=len(date_range)),
        name="target",
    )

    return {"X": X, "y": y}


@pytest.fixture
def simple_mmm_data():
    """Create simple single-dimension MMM data.

    Returns dict with:
    - X: DataFrame with date and 3 channels (14 weekly periods)
    - y: Series with target values
    """
    return _make_mmm_data(periods=14, freq="W", n_channels=3, seed=42)


@pytest.fixture
def monthly_mmm_data():
    """Create monthly-frequency (MS) MMM data for calendar-aware date math tests.

    Uses month-start frequency where months have variable lengths (28-31 days).
    This exposes the _convert_frequency_to_timedelta bug that approximates
    months as fixed 30-day periods.

    Returns dict with:
    - X: DataFrame with date and 2 channels (18 monthly periods)
    - y: Series with target values
    """
    return _make_mmm_data(periods=18, freq="MS", n_channels=3, seed=99)


@pytest.fixture
def log_link_mmm_data():
    """Create MMM data for a multiplicative (log-link) model.

    Two channels keep the per-channel counterfactual ground truth cheap, and
    the target is strictly positive as the log link requires.

    Returns dict with:
    - X: DataFrame with date and 2 channels (14 weekly periods)
    - y: Series with strictly positive target values
    """
    return _make_mmm_data(periods=14, freq="W", n_channels=2, seed=42)


@pytest.fixture
def panel_mmm_data():
    """Create panel (multidimensional) MMM data with country dimension.

    Returns dict with:
    - X: DataFrame with date, country, and 2 channels (7 periods × 2 countries)
    - y: Series with target values
    """
    date_range = pd.date_range("2023-01-01", periods=14, freq="W")
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
# Mock Fit Function
# ============================================================================


def mock_fit(model, X: pd.DataFrame, y: pd.Series, random_seed=None, **kwargs):
    """Mock fit function that mimics the fit process without actual sampling."""
    model.build_model(X=X, y=y)
    model.add_original_scale_contribution_variable(var=["channel_contribution"])

    # Explicitly set model data via pm.set_data so channel_data has concrete shape,
    # matching real mmm.fit() behavior. Without this, channel_data may retain
    # symbolic shape unlike mmm.fit().
    model._set_xarray_data(model.xarray_dataset, model=model.model)

    if random_seed is None:
        random_seed = rng
    with model.model:
        idata = pm.sample_prior_predictive(random_seed=random_seed, **kwargs)

    fit_data = model.create_fit_data(X, y)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message="The group fit_data is not defined in the DataTree scheme",
        )
        idata["/posterior"] = idata["/prior"].to_dataset()
        idata["/fit_data"] = fit_data
    model.idata = idata
    model.set_idata_attrs(idata=idata)

    return model


# ============================================================================
# Fitted Model Fixtures
# ============================================================================


@pytest.fixture
def simple_fitted_mmm(simple_mmm_data):
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

    mock_fit(mmm, X, y, random_seed=seed)

    return mmm


@pytest.fixture
def simple_fitted_mmm_int(simple_mmm_data):
    """Like simple_fitted_mmm but with float channel data (same values).

    Used to obtain correct marginal incrementality when factor produces
    fractional values (e.g. 1.01 * 50 = 50.5). Integer channel_data
    truncates; float does not. Uses same random seed as simple_fitted_mmm
    so posteriors match and the only difference is channel_data dtype.
    """
    X = simple_mmm_data["X"].copy()
    y = simple_mmm_data["y"]
    # Convert channel columns to float so counterfactual factor preserves fractions
    for col in ["channel_1", "channel_2", "channel_3"]:
        X[col] = X[col].astype(np.int64)

    mmm = MMM(
        channel_columns=["channel_1", "channel_2", "channel_3"],
        date_column="date",
        target_column="target",
        control_columns=None,
        adstock=GeometricAdstock(l_max=10),
        saturation=LogisticSaturation(),
    )

    # Use same seed as simple_fitted_mmm for comparable posteriors
    mock_fit(mmm, X, y, random_seed=seed)

    return mmm


@pytest.fixture
def monthly_fitted_mmm(monthly_mmm_data):
    """Create a monthly-frequency MMM with WeibullCDFAdstock."""
    X = monthly_mmm_data["X"]
    y = monthly_mmm_data["y"]

    mmm = MMM(
        channel_columns=["channel_1", "channel_2"],
        date_column="date",
        target_column="target",
        control_columns=None,
        adstock=WeibullCDFAdstock(l_max=3),
        saturation=LogisticSaturation(),
    )

    mock_fit(mmm, X, y)

    return mmm


@pytest.fixture
def time_varying_media_fitted_mmm(simple_mmm_data):
    """Create a fitted MMM with time_varying_media=True.

    This fixture produces a model whose channel_contribution graph contains
    the HSGP-based ``media_temporal_latent_multiplier``, which depends on the
    ``time_index`` shared variable (fixed at ``n_dates``).  It is used to
    test that incrementality evaluation correctly handles date-dependent
    latent tensors.
    """
    X = simple_mmm_data["X"]
    y = simple_mmm_data["y"]

    mmm = MMM(
        channel_columns=["channel_1", "channel_2", "channel_3"],
        date_column="date",
        target_column="target",
        control_columns=None,
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        time_varying_media=True,
    )

    mock_fit(mmm, X, y)
    mmm.post_sample_model_transformation()

    return mmm


@pytest.fixture
def time_varying_intercept_fitted_mmm(simple_mmm_data):
    """Create a fitted MMM with time_varying_intercept=True but time_varying_media=False.

    This combination means ``time_index`` exists in the model (for the
    HSGP intercept) but is **not** an ancestor of ``channel_contribution``.
    It reproduces the ``UnusedInputError`` reported in GH-XXXX when the
    incrementality code unconditionally added ``time_index`` as a compiled
    function input.
    """
    X = simple_mmm_data["X"]
    y = simple_mmm_data["y"]

    mmm = MMM(
        channel_columns=["channel_1", "channel_2", "channel_3"],
        date_column="date",
        target_column="target",
        control_columns=None,
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        time_varying_intercept=True,
        time_varying_media=False,
    )

    mock_fit(mmm, X, y)
    mmm.post_sample_model_transformation()

    return mmm


def _mock_fit_log_link(mmm, X, y):
    """Run ``mock_fit`` on a log-link model, silencing its two expected warnings.

    ``link="log"`` is flagged experimental, and ``mock_fit`` registers
    ``channel_contribution_original_scale``, which warns under a log link
    because a per-component ``*_original_scale`` variable is a multiplicative
    factor rather than an additive contribution.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mock_fit(mmm, X, y, random_seed=seed)
    return mmm


@pytest.fixture
def log_link_fitted_mmm(log_link_mmm_data):
    """Create a fitted multiplicative (log-link) MMM.

    Pairs ``link="log"`` with :class:`LogSaturation`, the usual specification
    for a multiplicative model.  ``LogSaturation`` maps zero spend to zero
    contribution, which the total-media invariant test relies on.
    """
    mmm = MMM(
        channel_columns=["channel_1", "channel_2"],
        date_column="date",
        target_column="target",
        control_columns=None,
        adstock=GeometricAdstock(l_max=4),
        saturation=LogSaturation(),
        link="log",
    )

    return _mock_fit_log_link(mmm, log_link_mmm_data["X"], log_link_mmm_data["y"])


@pytest.fixture
def log_link_single_channel_fitted_mmm(log_link_mmm_data):
    """Create a fitted log-link MMM with exactly one channel.

    With a single channel the media term of the linear predictor *is* that
    channel's contribution, so its incremental contribution must equal the
    model's own ``total_media_contribution_original_scale``.
    """
    mmm = MMM(
        channel_columns=["channel_1"],
        date_column="date",
        target_column="target",
        control_columns=None,
        adstock=GeometricAdstock(l_max=4),
        saturation=LogSaturation(),
        link="log",
    )

    return _mock_fit_log_link(mmm, log_link_mmm_data["X"], log_link_mmm_data["y"])


@pytest.fixture
def log_link_control_fitted_mmm(log_link_mmm_data):
    """Create a fitted log-link MMM with a control variable.

    Controls sit in the baseline, and under a log link the baseline *weights*
    the increment instead of cancelling out of it, so a channel's incremental
    contribution depends on them.  Pairs with the per-channel oracle to check
    that dependence is handled rather than assumed away.
    """
    X = log_link_mmm_data["X"].copy()
    X["control_1"] = np.linspace(-1.0, 1.0, len(X))

    mmm = MMM(
        channel_columns=["channel_1", "channel_2"],
        date_column="date",
        target_column="target",
        control_columns=["control_1"],
        adstock=GeometricAdstock(l_max=4),
        saturation=LogSaturation(),
        link="log",
    )

    return _mock_fit_log_link(mmm, X, log_link_mmm_data["y"])


@pytest.fixture
def log_link_time_varying_media_fitted_mmm(log_link_mmm_data):
    """Create a fitted log-link MMM with ``time_varying_media=True``.

    ``channel_contribution`` then carries the HSGP
    ``media_temporal_latent_multiplier``, which depends on ``time_index``, so
    this combines the log-link reduction with the module's date-dependent
    evaluation path.
    """
    mmm = MMM(
        channel_columns=["channel_1", "channel_2"],
        date_column="date",
        target_column="target",
        control_columns=None,
        adstock=GeometricAdstock(l_max=4),
        saturation=LogSaturation(),
        link="log",
        time_varying_media=True,
    )

    mmm = _mock_fit_log_link(mmm, log_link_mmm_data["X"], log_link_mmm_data["y"])
    mmm.post_sample_model_transformation()

    return mmm


@pytest.fixture
def log_link_panel_fitted_mmm(panel_mmm_data):
    """Create a fitted panel (multidimensional) log-link MMM.

    Exercises the broadcast between the baseline response, which carries
    ``(date, country)``, and the perturbation, which carries
    ``(date, channel, country)``.
    """
    X = panel_mmm_data["X"].copy()
    for col in X.columns[X.columns.str.startswith("channel_")]:
        X[col] = X[col].astype(np.float64)

    mmm = MMM(
        channel_columns=["channel_1", "channel_2"],
        date_column="date",
        target_column="target",
        dims=("country",),
        control_columns=None,
        adstock=GeometricAdstock(l_max=4),
        saturation=LogSaturation(),
        link="log",
    )

    return _mock_fit_log_link(mmm, X, panel_mmm_data["y"])


@pytest.fixture
def panel_fitted_mmm(panel_mmm_data):
    """Create a panel (multidimensional) fitted MMM for testing."""
    X = panel_mmm_data["X"]
    y = panel_mmm_data["y"]

    # Convert channel columns to float so we could test on a model with float channel_data
    channel_columns = X.columns[X.columns.str.startswith("channel_")]
    for col in channel_columns:
        X[col] = X[col].astype(np.float64)

    adstock = GeometricAdstock(
        priors={"alpha": Prior("Beta", alpha=2, beta=5, dims=("country", "channel"))},
        l_max=10,
    )

    beta_prior = LogNormalPrior(
        mean=Prior("Gamma", mu=0.25, sigma=0.10, dims=("channel")),
        std=Prior("Exponential", scale=0.10, dims=("channel")),
        dims=("channel", "country"),
        centered=False,
    )
    saturation = LogisticSaturation(
        priors={
            "beta": beta_prior,
            "lam": Prior(
                "Gamma",
                mu=0.5,
                sigma=0.25,
                dims=("channel"),
            ),
        }
    )
    mmm = MMM(
        channel_columns=["channel_1", "channel_2"],
        date_column="date",
        target_column="target",
        dims=("country",),
        control_columns=None,
        adstock=adstock,
        saturation=saturation,
        yearly_seasonality=2,
    )

    mock_fit(mmm, X, y)

    return mmm
