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
import pymc.dims as pmd
import pytest
import xarray as xr
from pydantic import InstanceOf
from pymc_extras.prior import Prior
from pytensor.graph.basic import Variable
from pytensor.graph.traversal import ancestors

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation, LogSaturation
from pymc_marketing.mmm.additive_effect import (
    DataVarMuEffect,
    IncrementalitySpec,
    LinearTrendEffect,
    MuEffect,
)
from pymc_marketing.mmm.components.adstock import WeibullCDFAdstock
from pymc_marketing.mmm.components.saturation import SaturationTransformation
from pymc_marketing.mmm.linear_trend import LinearTrend
from pymc_marketing.mmm.media_transformation import MediaTransformation
from pymc_marketing.mmm.mmm import MMM
from pymc_marketing.serialization import serialization
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


@pytest.fixture(scope="module")
def log_link_mmm_data():
    """Create MMM data for a multiplicative (log-link) model.

    Two channels keep the per-channel counterfactual ground truth cheap, and
    the target is strictly positive as the log link requires.

    Returns dict with:
    - X: DataFrame with date and 2 channels (14 weekly periods)
    - y: Series with strictly positive target values
    """
    return _make_mmm_data(periods=14, freq="W", n_channels=2, seed=42)


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
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


def _split_posterior_into_two_chains(mmm):
    """Re-lay the fitted posterior's draws out as two chains instead of one.

    ``mock_fit`` stands a prior-predictive sample in for a posterior, and that
    always arrives with ``chain=1``.  Incrementality flattens ``(chain, draw)``
    into a single ``sample`` axis to line the posterior up with a batched graph
    evaluation, and with one chain that flattening is a no-op -- so a mismatch
    between its ordering and ``arviz``'s would be invisible, and would show up as
    a silently wrong number rather than an error.  Splitting the same draws
    across two chains makes the ordering observable while leaving every value
    unchanged, so the oracle stays comparable.

    The tree is rebuilt rather than patched in place: the ``prior`` groups the
    stand-in fit leaves behind keep the original draw count, and a ``DataTree``
    will not hold two sizes for one dimension.  They are of no use to anything
    downstream, so they go.
    """
    posterior = mmm.idata["/posterior"].to_dataset()
    half = posterior.sizes["draw"] // 2

    def chain(start):
        return posterior.isel(chain=0, draw=slice(start, start + half)).assign_coords(
            draw=np.arange(half)
        )

    groups = {
        "/posterior": xr.concat(
            [chain(0), chain(half)], dim="chain", join="exact"
        ).assign_coords(chain=[0, 1])
    }
    for name in ("fit_data", "constant_data", "observed_data"):
        if name in mmm.idata.children:
            groups[f"/{name}"] = mmm.idata[name].to_dataset()

    idata = xr.DataTree.from_dict(groups)
    mmm.idata = idata
    mmm.set_idata_attrs(idata=idata)
    return mmm


@pytest.fixture(scope="module")
def log_link_two_chain_fitted_mmm(log_link_mmm_data):
    """A log-link MMM whose posterior carries two chains rather than one."""
    mmm = MMM(
        channel_columns=["channel_1", "channel_2"],
        date_column="date",
        target_column="target",
        control_columns=None,
        adstock=GeometricAdstock(l_max=4),
        saturation=LogSaturation(),
        link="log",
    )
    _mock_fit_log_link(mmm, log_link_mmm_data["X"], log_link_mmm_data["y"])
    return _split_posterior_into_two_chains(mmm)


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
def log_link_fixed_sigma_fitted_mmm(log_link_mmm_data):
    """Log-link MMM whose LogNormal sigma is a fixed number rather than a prior.

    With no ``y_sigma`` in the posterior there is nothing to build the
    LogNormal mean correction from, so ``central_tendency="mean"`` has to fail
    with a clear message instead of silently returning median-scale numbers.
    """
    mmm = MMM(
        channel_columns=["channel_1", "channel_2"],
        date_column="date",
        target_column="target",
        control_columns=None,
        adstock=GeometricAdstock(l_max=4),
        saturation=LogSaturation(),
        link="log",
        model_config={"likelihood": Prior("LogNormal", sigma=0.5, dims="date")},
    )

    return _mock_fit_log_link(mmm, log_link_mmm_data["X"], log_link_mmm_data["y"])


@pytest.fixture
def panel_fitted_mmm(panel_mmm_data):
    """Create a panel (multidimensional) fitted MMM for testing."""
    # Copy: ``panel_mmm_data`` is module-scoped and the dtype cast below must
    # not leak into other consumers of the shared frame.
    X = panel_mmm_data["X"].copy()
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


# ============================================================================
# Funnel (mediated) MuEffect Fixtures
# ============================================================================
#
# A funnel effect is the motivating case for incrementality that reaches the
# response through a ``mu_effect``: upper-funnel spend moves latent demand,
# demand moves lower-funnel spend, and only then does the target respond.  Two
# properties make it the interesting test case rather than just another effect:
#
# 1. It reads ``channel_data`` (through ``mmm.channel_data_scaled``), so a spend
#    counterfactual moves it.  Ignoring it reports the direct path alone.
# 2. It sums the channel dimension away *inside* a nonlinear transform, so no
#    per-channel column survives to be read off a single all-channels
#    counterfactual.  One counterfactual per channel becomes mandatory.
#
# It also brings its own date-indexed ``pm.Data`` (``lf_budget``) and chains a
# second adstock behind the model's own, which is what the evaluation window has
# to be widened for.  Mirrors the ``mmm_funnel_mueffect`` example notebook.


class FunnelEffect(DataVarMuEffect):
    """Upper-funnel spend -> latent demand -> lower-funnel spend -> target.

    Parameters
    ----------
    upper_transform : MediaTransformation
        Adstock/saturation applied to channel spend before it enters demand.
    demand_transform : MediaTransformation
        Adstock/saturation applied to the mediator before it reaches the target.
        Its ``l_max`` is the extra carryover the effect adds.
    """

    upper_transform: InstanceOf[MediaTransformation]
    demand_transform: InstanceOf[MediaTransformation]

    model_config = {"arbitrary_types_allowed": True}

    def to_dict(self) -> dict:
        """Serialize the effect."""
        return {
            "data_vars": self.data_vars,
            "prefix": self.prefix,
            "upper_transform": self.upper_transform.to_dict(),
            "demand_transform": self.demand_transform.to_dict(),
        }

    def incrementality_spec(self) -> IncrementalitySpec:
        """Opt in, declaring the second adstock's reach."""
        return IncrementalitySpec(
            additional_carryover_lags=self.demand_transform.adstock.l_max
        )

    def create_effect(self, mmm):
        """Build the mediator equation and the term it adds to the target mean."""
        model = mmm.model
        # (date, channel) -> (date): every channel pushes the same demand pool,
        # so the channel dimension is summed away inside the transform.
        upper_on_demand = self.upper_transform(mmm.channel_data_scaled, dim="date").sum(
            dim="channel"
        )
        baseline = pmd.HalfNormal(f"{self.prefix}_baseline", sigma=1.0)
        demand = pmd.Deterministic(f"{self.prefix}_demand", baseline + upper_on_demand)
        lam = pmd.HalfNormal(f"{self.prefix}_lambda", sigma=0.5)
        # Lower-funnel spend responds to demand and to its own exogenous budget,
        # the latter a date-indexed pm.Data the evaluation has to window too.
        lf_spend = pmd.Deterministic(
            f"{self.prefix}_lf_spend", demand + lam * model["lf_budget"]
        )
        return pmd.Deterministic(
            f"{self.prefix}_effect_contribution",
            self.demand_transform(lf_spend, dim="date"),
        )


class UnregisteredFunnelEffect(FunnelEffect):
    """A funnel effect that has *not* opted in to incrementality."""

    def incrementality_spec(self) -> None:
        """Opt out, so incrementality has to refuse rather than guess."""
        return None


class UnderDeclaredFunnelEffect(FunnelEffect):
    """A funnel effect that claims to add no carryover of its own.

    Its second adstock plainly does add some, so the declaration is false and
    the evaluation window built from it would cut the mediated tail off.  Which
    is precisely the failure the measurement exists to catch.
    """

    def incrementality_spec(self) -> IncrementalitySpec:
        """Declare a reach the effect demonstrably exceeds."""
        return IncrementalitySpec(additional_carryover_lags=0)


class PartialChannelFunnelEffect(FunnelEffect):
    """A funnel effect fed by the first channel alone, with nothing declared.

    Reading a subset of channels is what makes probe *placement* matter: a probe
    at a date where that channel is dark cannot move this effect, so its reach
    would be measured as nothing at all.  Declaring nothing -- the documented
    "measure it" default -- makes that measurement the only thing the window has
    to go on, so a missed probe cuts the mediated tail silently.
    """

    def incrementality_spec(self) -> IncrementalitySpec:
        """Opt in and declare nothing: the reach is measured."""
        return IncrementalitySpec()

    def create_effect(self, mmm):
        """Build the mediator equation off the first channel only."""
        model = mmm.model
        # (date, channel) -> (date): only the first channel feeds demand.
        upper_on_demand = self.upper_transform(
            mmm.channel_data_scaled, dim="date"
        ).isel(channel=0)
        baseline = pmd.HalfNormal(f"{self.prefix}_baseline", sigma=1.0)
        demand = pmd.Deterministic(f"{self.prefix}_demand", baseline + upper_on_demand)
        lam = pmd.HalfNormal(f"{self.prefix}_lambda", sigma=0.5)
        lf_spend = pmd.Deterministic(
            f"{self.prefix}_lf_spend", demand + lam * model["lf_budget"]
        )
        return pmd.Deterministic(
            f"{self.prefix}_effect_contribution",
            self.demand_transform(lf_spend, dim="date"),
        )


class MisreportedFunnelEffect(FunnelEffect):
    """A funnel effect pointing at a variable that is not its contribution.

    ``funnel_demand`` is a real variable of the model, so nothing about the name
    looks wrong, and it even depends on spend -- but the term the effect adds to
    the linear predictor is ``funnel_effect_contribution``, which then goes
    unevaluated.
    """

    @property
    def contribution_var_name(self) -> str:
        """Name a real, spend-dependent, but wrong variable."""
        return f"{self.prefix}_demand"


class FullAxisFunnelEffect(FunnelEffect):
    """A funnel effect that asks for the full date axis it does not need.

    Its reach is bounded and measurable, so ``evaluation_mode="auto"`` would give
    it a window.  Declaring ``"full"`` has to override that: an effect's author
    may know of a dependence on the whole series the probe's single perturbation
    cannot show, and the cost of honouring the declaration is compute rather than
    correctness.
    """

    def incrementality_spec(self) -> IncrementalitySpec:
        """Opt in, declare the reach, and insist on the full axis anyway."""
        return IncrementalitySpec(
            additional_carryover_lags=self.demand_transform.adstock.l_max,
            evaluation_mode="full",
        )


class LongTailFunnelEffect(FunnelEffect):
    """A funnel effect whose mediated tail outlives the date axis itself.

    Same graph as :class:`FunnelEffect`, but the second adstock is long enough
    relative to the series that a perturbation early on still moves the *last*
    fitted date.  The axis can then show only that the tail has not ended, never
    how long it is, so full-axis evaluation is selected: exactly the case in
    which the carry-out entering each period's sum must not be cut back to the
    model's own ``l_max``.

    It declares nothing, which is the documented "measure it" default, so the
    measurement is the only thing the evaluation has to go on.
    """

    def incrementality_spec(self) -> IncrementalitySpec:
        """Opt in and declare nothing: the reach is measured."""
        return IncrementalitySpec()


class GlobalNormalizationEffect(MuEffect):
    """An effect that divides spend by its own mean over the whole date axis.

    Nothing about it is causal in time, and that is the point: its value at any
    date depends on *every* date, so evaluating it on a truncated window gives a
    different -- wrong -- answer.  It has to force full-axis evaluation, and the
    only way to know that without being told is to measure it.
    """

    prefix: str

    def to_dict(self) -> dict:
        """Serialize the effect."""
        return {"prefix": self.prefix}

    def create_data(self, mmm) -> None:
        """No data of its own to register."""

    def set_data(self, mmm, model, X) -> None:
        """No data of its own to update."""

    def incrementality_spec(self) -> IncrementalitySpec:
        """Opt in and declare nothing: the mode is measured."""
        return IncrementalitySpec()

    def create_effect(self, mmm):
        """Add spend normalized by its own date-mean to the linear predictor."""
        spend = mmm.channel_data_scaled.sum(dim="channel")
        beta = pmd.HalfNormal(f"{self.prefix}_beta", sigma=1.0)
        return pmd.Deterministic(
            f"{self.prefix}_effect_contribution",
            beta * spend / spend.mean(dim="date"),
        )


class WindowedGlobalNormalizationEffect(GlobalNormalizationEffect):
    """A date-reducing effect that insists a window is enough for it.

    The declaration is false, and falsifiably so: the probe measures the effect
    moving dates before the perturbed one.  Honouring it would evaluate a
    different function -- the mean of the window rather than of the series -- so
    it has to be refused rather than accepted or silently widened.
    """

    def incrementality_spec(self) -> IncrementalitySpec:
        """Declare a mode the effect demonstrably cannot be evaluated in."""
        return IncrementalitySpec(evaluation_mode="window")


class DuckTypedEffect:
    """A ``mu_effect`` that is not a :class:`MuEffect` at all.

    The additive-effect module documents this pattern -- anything with
    ``create_effect`` and ``set_data`` will do -- so incrementality cannot
    require ``contribution_var_name`` of every effect it meets.  Reading spend
    is optional and controlled by *reads_spend*, because the two cases have to
    end differently: a baseline effect is none of incrementality's business,
    while one that moves with spend is an unaccounted path and must be refused.
    """

    def __init__(self, prefix: str, reads_spend: bool) -> None:
        self.prefix = prefix
        self.reads_spend = reads_spend

    def to_dict(self) -> dict:
        """Serialize the effect, which fitting insists on."""
        return {"prefix": self.prefix, "reads_spend": self.reads_spend}

    def create_effect(self, mmm):
        """Add a term that reads spend, or does not."""
        beta = pmd.HalfNormal(f"{self.prefix}_beta", sigma=1.0)
        if not self.reads_spend:
            return beta
        return beta * mmm.channel_data_scaled.sum(dim="channel")

    def create_data(self, mmm) -> None:
        """No data of its own to register."""

    def set_data(self, mmm, model, X) -> None:
        """No data of its own to update."""


serialization.register(
    f"{DuckTypedEffect.__module__}.{DuckTypedEffect.__qualname__}",
    DuckTypedEffect,
    deserializer=lambda data: DuckTypedEffect(**data),
)


class ChainedMediatorEffect(MuEffect):
    """A funnel-shaped mediator that brings no data of its own.

    Same shape of dependence as :class:`FunnelEffect` -- two adstocks in series
    behind the model's own -- but without the ``lf_budget`` input, so a model can
    carry two of these at once without two effects trying to register the same
    ``pm.Data``.

    Parameters
    ----------
    prefix : str
        Names the effect's variables.
    upper_transform, demand_transform : MediaTransformation
        The two transforms spend passes through.  ``demand_transform``'s
        ``l_max`` is the extra carryover the effect adds.
    """

    prefix: str
    upper_transform: InstanceOf[MediaTransformation]
    demand_transform: InstanceOf[MediaTransformation]

    model_config = {"arbitrary_types_allowed": True}

    def to_dict(self) -> dict:
        """Serialize the effect."""
        return {
            "prefix": self.prefix,
            "upper_transform": self.upper_transform.to_dict(),
            "demand_transform": self.demand_transform.to_dict(),
        }

    def create_data(self, mmm) -> None:
        """No data of its own to register."""

    def set_data(self, mmm, model, X) -> None:
        """No data of its own to update."""

    def incrementality_spec(self) -> IncrementalitySpec:
        """Opt in, declaring the second adstock's reach."""
        return IncrementalitySpec(
            additional_carryover_lags=self.demand_transform.adstock.l_max
        )

    def create_effect(self, mmm):
        """Chain a second transform behind the first, as a funnel does."""
        demand = self.upper_transform(mmm.channel_data_scaled, dim="date").sum(
            dim="channel"
        )
        return pmd.Deterministic(
            f"{self.prefix}_effect_contribution",
            self.demand_transform(demand, dim="date"),
        )


class SubsetDimsEffect(MuEffect):
    """A spend-dependent effect whose contribution drops a model dimension.

    Sums the panel dimension away, so its contribution carries ``("date",)``
    where the model carries ``("date", "country")``.  Legitimate -- a term added
    to ``mu`` may broadcast -- and worth pinning, because the increment sums the
    effect's delta into a per-channel array that does carry the panel dimension.
    """

    prefix: str

    def to_dict(self) -> dict:
        """Serialize the effect."""
        return {"prefix": self.prefix}

    def create_data(self, mmm) -> None:
        """No data of its own to register."""

    def set_data(self, mmm, model, X) -> None:
        """No data of its own to update."""

    def incrementality_spec(self) -> IncrementalitySpec:
        """Opt in; the effect is instantaneous, so nothing needs declaring."""
        return IncrementalitySpec()

    def create_effect(self, mmm):
        """Add total spend, summed over every dimension but ``date``."""
        spend = mmm.channel_data_scaled
        beta = pmd.HalfNormal(f"{self.prefix}_beta", sigma=1.0)
        return pmd.Deterministic(
            f"{self.prefix}_effect_contribution",
            beta * spend.sum(dim=[d for d in spend.dims if d != "date"]),
        )


class NegativeEffect(MuEffect):
    """A spend-dependent effect that *subtracts* from the linear predictor.

    Cannibalisation, in marketing terms: one channel's spend eats into the
    response.  Its point here is the sign.  The claim that per-channel
    increments sum to more than the joint one rests on every channel's
    contribution being positive; with a negative mediated term the interaction
    can go the other way, and an estimand docstring that asserted a direction
    would be wrong.
    """

    prefix: str

    def to_dict(self) -> dict:
        """Serialize the effect."""
        return {"prefix": self.prefix}

    def create_data(self, mmm) -> None:
        """No data of its own to register."""

    def set_data(self, mmm, model, X) -> None:
        """No data of its own to update."""

    def incrementality_spec(self) -> IncrementalitySpec:
        """Opt in; the effect is instantaneous, so nothing needs declaring."""
        return IncrementalitySpec()

    def create_effect(self, mmm):
        """Subtract a multiple of the product of the channels' spend."""
        spend = mmm.channel_data_scaled
        beta = pmd.HalfNormal(f"{self.prefix}_beta", sigma=2.0)
        return pmd.Deterministic(
            f"{self.prefix}_effect_contribution",
            -beta * spend.prod(dim="channel"),
        )


for _effect_cls in (ChainedMediatorEffect, SubsetDimsEffect, NegativeEffect):
    serialization.register(
        f"{_effect_cls.__module__}.{_effect_cls.__qualname__}",
        _effect_cls,
        deserializer=lambda data, cls=_effect_cls: cls(**data),
    )


def _media_transformation(prefix, dims, l_max, beta_sigma=1.0):
    """Adstock/saturation pair with priors carrying *dims*.

    ``beta_sigma`` scales the transform's output.  The funnel's two transforms
    compose, so each one's saturation damps the next one's sensitivity to spend;
    without a wider prior the mediated path ends up a rounding correction on the
    direct one, and a test asserting the path is included would have nothing to
    fail on.
    """
    dim = dims[0] if dims else None
    return MediaTransformation(
        adstock=GeometricAdstock(
            l_max=l_max,
            prefix=f"adstock_{prefix}",
            priors={"alpha": Prior("Beta", alpha=2, beta=3, dims=dim)},
        ),
        saturation=LogisticSaturation(
            prefix=f"saturation_{prefix}",
            priors={
                "lam": Prior("HalfNormal", sigma=1.0, dims=dim),
                "beta": Prior("HalfNormal", sigma=beta_sigma, dims=dim),
            },
        ),
        dims=dims,
        adstock_first=True,
    )


FUNNEL_L_MAX = 3
"""Adstock length used by both the model and the funnel's second transform."""


@pytest.fixture(scope="module")
def funnel_mmm_data():
    """Spend, an exogenous lower-funnel budget, and a strictly positive target.

    Two shapes of the same data, as the funnel example notebook uses: an
    ``xr.Dataset`` to build the model from, because the effect's ``lf_budget``
    only exists there, and a plain frame of date plus channels for ``fit_data``,
    whose long-form conversion cannot align a separate ``y`` against a variable
    carrying a ``channel`` dimension.
    """
    n_dates = 24
    local_rng = np.random.default_rng(20260804)
    dates = pd.date_range("2023-01-02", freq="W-MON", periods=n_dates)
    media = np.abs(local_rng.normal(1.0, 0.35, size=(n_dates, 2))) + 0.2
    lf_budget = np.abs(local_rng.normal(0.8, 0.2, size=n_dates))
    target = np.exp(1.0 + 0.4 * media[:, 0] + 0.3 * media[:, 1] + 0.25 * lf_budget)
    channels = ["channel_1", "channel_2"]

    X = xr.Dataset(
        {
            "media": xr.DataArray(media, dims=("date", "channel")),
            "lf_budget": xr.DataArray(lf_budget, dims=("date",)),
        },
        coords={"date": dates, "channel": channels},
    )
    y = xr.DataArray(target, dims=("date",), coords={"date": dates})
    X_df = pd.DataFrame({"date": dates, **dict(zip(channels, media.T, strict=True))})
    return {"X": X, "y": y, "X_df": X_df, "y_series": pd.Series(target, name="y")}


def _funnel_effect(effect_cls=FunnelEffect, *, prefix="funnel", demand_l_max=None):
    """A funnel effect whose second adstock reaches *demand_l_max* periods."""
    return effect_cls(
        data_vars=["lf_budget"],
        prefix=prefix,
        upper_transform=_media_transformation(
            f"upper_{prefix}", ("channel",), FUNNEL_L_MAX, beta_sigma=3.0
        ),
        demand_transform=_media_transformation(
            f"demand_{prefix}",
            (),
            FUNNEL_L_MAX if demand_l_max is None else demand_l_max,
            beta_sigma=3.0,
        ),
    )


def _build_funnel_mmm(
    data,
    link,
    effect_cls=FunnelEffect,
    *,
    effects=None,
    l_max=FUNNEL_L_MAX,
    time_varying_media=False,
    saturation=None,
):
    """Fit a funnel MMM under *link* with prior draws standing in for a posterior.

    Mirrors the funnel example notebook: build from the ``xr.Dataset`` so the
    effect's data variables register, then create ``fit_data`` from the frame.
    ``lf_budget`` therefore never reaches ``fit_data``, which is why the evaluator
    reads auxiliary inputs from the model's own shared variables.

    Parameters
    ----------
    data : dict
        The ``funnel_mmm_data`` fixture.
    link : str
        Link function name.
    effect_cls : type, default=FunnelEffect
        Which funnel effect to attach.  Ignored when *effects* is given.
    effects : list, optional
        Effects to attach instead of a single default funnel effect.
    l_max : int, default=FUNNEL_L_MAX
        The model's *own* adstock length.  Zero is worth testing: it leaves the
        mediated path as the only source of carryover.
    time_varying_media : bool, default=False
        Add the HSGP media multiplier, which puts ``time_index`` in the graph and
        so makes the window's position on the date axis matter.
    saturation : SaturationTransformation, optional
        The model's own saturation, defaulting to a per-channel logistic one.
        Overridden by the mixing fixtures, whose whole point is a media transform
        that is *not* separable across channels.
    """
    mmm = MMM(
        channel_columns=["channel_1", "channel_2"],
        date_column="date",
        target_column="y",
        control_columns=None,
        adstock=GeometricAdstock(l_max=l_max),
        saturation=LogisticSaturation() if saturation is None else saturation,
        time_varying_media=time_varying_media,
        link=link,
    )
    for effect in effects if effects is not None else [_funnel_effect(effect_cls)]:
        mmm.add_mu_effect(effect)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mmm.build_model(X=data["X"], y=data["y"])
        mmm.add_original_scale_contribution_variable(var=["channel_contribution"])
        mmm._set_xarray_data(mmm.xarray_dataset, model=mmm.model)
        with mmm.model:
            # A small draw count keeps the per-(period, channel) oracle these
            # fixtures are checked against affordable; the algebra under test does
            # not depend on how many draws there are.
            idata = pm.sample_prior_predictive(draws=50, random_seed=seed)
        idata["/posterior"] = idata["/prior"].to_dataset()
        idata["/fit_data"] = mmm.create_fit_data(data["X_df"], data["y_series"])
    mmm.idata = idata
    mmm.set_idata_attrs(idata=idata)
    return mmm


@serialization.register
class SharedDenominatorSaturation(SaturationTransformation):
    r"""A media transform that is *not* separable across channels.

    Channels compete for one pool of attention, so every channel's contribution
    carries every other channel's spend in its denominator:

    .. math::

        v_c = \beta \frac{x_c}{1 + \sum_{c'} x_{c'}}

    Nothing in the MMM forbids this -- ``forward_pass`` hands the saturation the
    whole ``(date, channel)`` tensor -- and it breaks the assumption the separable
    readout rests on: column *m* of an all-channels counterfactual is no longer
    channel *m*'s unilateral counterfactual, because zeroing channel *m* alone
    also *raises* every other column.
    """

    def function(self, x, beta, *, dim: str | None = None):
        """Saturate each channel against the total spend of all of them."""
        return beta * x / (1.0 + x.sum("channel"))

    default_priors = {"beta": Prior("HalfNormal", sigma=1.0)}


@pytest.fixture(scope="module")
def mixing_identity_fitted_mmm(funnel_mmm_data):
    """Additive MMM whose media transform mixes channels, with no ``mu_effects``.

    The separable branch of the increment readout: with no channel-dependent
    effect there is nothing to force per-channel scenarios, so the mixing has to
    be *measured* or the per-channel numbers come off a joint perturbation.
    """
    mmm = MMM(
        channel_columns=["channel_1", "channel_2"],
        date_column="date",
        target_column="y",
        control_columns=None,
        adstock=GeometricAdstock(l_max=FUNNEL_L_MAX),
        saturation=SharedDenominatorSaturation(),
        link="identity",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mock_fit(
            mmm,
            funnel_mmm_data["X_df"],
            funnel_mmm_data["y_series"],
            random_seed=seed,
            draws=50,
        )
    return mmm


@pytest.fixture(scope="module")
def mixing_funnel_identity_fitted_mmm(funnel_mmm_data):
    """Additive funnel MMM whose media transform also mixes channels.

    The mediated branch: the funnel effect already forces one counterfactual per
    channel, and the mixing is what makes the *readout* of those counterfactuals
    matter -- the other channels' contribution columns move too, and dropping
    them understates the unilateral increment.
    """
    return _build_funnel_mmm(
        funnel_mmm_data, link="identity", saturation=SharedDenominatorSaturation()
    )


@pytest.fixture(scope="module")
def funnel_log_link_fitted_mmm(funnel_mmm_data):
    """Multiplicative funnel MMM: nonlinear link *and* a mediated path."""
    return _build_funnel_mmm(funnel_mmm_data, link="log")


@pytest.fixture(scope="module")
def funnel_identity_fitted_mmm(funnel_mmm_data):
    """Additive funnel MMM: the mediated path without the link nonlinearity."""
    return _build_funnel_mmm(funnel_mmm_data, link="identity")


@pytest.fixture(scope="module")
def funnel_not_opted_in_fitted_mmm(funnel_mmm_data):
    """Funnel MMM whose effect never opted in to incrementality."""
    return _build_funnel_mmm(
        funnel_mmm_data, link="log", effect_cls=UnregisteredFunnelEffect
    )


@pytest.fixture(scope="module")
def funnel_under_declared_fitted_mmm(funnel_mmm_data):
    """Funnel MMM whose effect declares less carryover than it has."""
    return _build_funnel_mmm(
        funnel_mmm_data, link="log", effect_cls=UnderDeclaredFunnelEffect
    )


@pytest.fixture(scope="module")
def funnel_misreported_fitted_mmm(funnel_mmm_data):
    """Funnel MMM whose effect names the wrong contribution variable."""
    return _build_funnel_mmm(
        funnel_mmm_data, link="log", effect_cls=MisreportedFunnelEffect
    )


@pytest.fixture(scope="module")
def funnel_instantaneous_adstock_fitted_mmm(funnel_mmm_data):
    """Funnel MMM whose own adstock carries nothing (``l_max=1``, lag zero only).

    Without the mediated path the evaluation window would collapse to the period
    itself, so this isolates the mediated carryover as the sole reason the window
    is any wider.
    """
    return _build_funnel_mmm(funnel_mmm_data, link="log", l_max=1)


@pytest.fixture(scope="module")
def funnel_full_axis_fitted_mmm(funnel_mmm_data):
    """Funnel MMM whose effect declares ``evaluation_mode="full"``."""
    return _build_funnel_mmm(
        funnel_mmm_data, link="log", effect_cls=FullAxisFunnelEffect
    )


LONG_TAIL_DEMAND_L_MAX = 22
"""Mediator adstock length in :func:`long_tail_funnel_fitted_mmm`.

The series is 24 dates long and the probe lands in its first few dates, so two
chained adstocks of 3 and 22 reach the very last one: the tail cannot be bounded
from the axis, which is what selects full-axis evaluation.
"""


@pytest.fixture(scope="module")
def long_tail_funnel_fitted_mmm(funnel_mmm_data):
    """Funnel MMM whose mediated tail still moves the last fitted date."""
    return _build_funnel_mmm(
        funnel_mmm_data,
        link="log",
        effects=[
            _funnel_effect(LongTailFunnelEffect, demand_l_max=LONG_TAIL_DEMAND_L_MAX)
        ],
    )


@pytest.fixture(scope="module")
def dark_start_funnel_mmm_data(funnel_mmm_data):
    """Funnel data whose first third carries no spend at all.

    A flighted campaign, a seasonal blackout, or a panel cell that starts late:
    every one of them puts a run of all-zero rows at the front of
    ``channel_data``.  The point is what that does to a *probe*.  Perturbing
    spend at an all-zero date multiplicatively changes nothing, so every node
    comes back unmoved, the measured reach is zero and the completeness check
    compares zero against zero -- both guards pass by seeing nothing.  A probe
    that picks its date by position rather than by spend walks straight into it.

    The spike at :data:`DARK_START` is where a probe that looks at spend will
    land, and it is early enough in the remaining series for the mediated tail to
    end before the last date, so the reach comes out bounded and the assertion can
    be a number.
    """
    data = {key: value.copy(deep=True) for key, value in funnel_mmm_data.items()}
    data["X"]["media"][:DARK_START] = 0.0
    data["X"]["media"][DARK_START] *= 4.0
    media = data["X"]["media"].values
    for i, channel in enumerate(["channel_1", "channel_2"]):
        data["X_df"][channel] = media[:, i]
    return data


DARK_START = 9
"""First date carrying spend in :func:`dark_start_funnel_mmm_data`."""


@pytest.fixture(scope="module")
def dark_start_funnel_fitted_mmm(dark_start_funnel_mmm_data):
    """Funnel MMM fit on data whose first third carries no spend."""
    return _build_funnel_mmm(dark_start_funnel_mmm_data, link="log")


@pytest.fixture(scope="module")
def partial_channel_funnel_fitted_mmm(funnel_mmm_data):
    """Funnel MMM whose mediator reads only ``channel_1`` and declares nothing.

    The control arm for :func:`dark_probe_funnel_fitted_mmm`: same graph, but
    every date carries spend on both channels, so any probe moves the mediated
    channel and the measured reach is whatever the graph truly does.
    """
    return _build_funnel_mmm(
        funnel_mmm_data, link="log", effect_cls=PartialChannelFunnelEffect
    )


@pytest.fixture(scope="module")
def dark_probe_funnel_mmm_data(funnel_mmm_data):
    """Funnel data whose two channels never spend on the same date.

    ``channel_1`` is dark until :data:`DARK_PROBE_SPLIT` and ``channel_2`` from
    then on -- a launch handover, or two flighted campaigns in sequence.  The
    point is what that does to a probe placed by *total* spend: every early date
    carries ``channel_2`` spend, so a single probe lands where ``channel_1`` is
    dark, and an effect reading only ``channel_1`` comes back bit-identical --
    measured reach nothing, mediated tail silently cut.  With no date moving both
    channels, only one probe per channel can cover them.

    The spike at :data:`DARK_PROBE_SPLIT` is where ``channel_1``'s own probe will
    land, early enough in the remaining series for the mediated tail to end
    before the last date, so the reach comes out bounded.
    """
    data = {key: value.copy(deep=True) for key, value in funnel_mmm_data.items()}
    media = data["X"]["media"].values
    media[:DARK_PROBE_SPLIT, 0] = 0.0
    media[DARK_PROBE_SPLIT:, 1] = 0.0
    media[DARK_PROBE_SPLIT, 0] = 8.0
    for i, channel in enumerate(["channel_1", "channel_2"]):
        data["X_df"][channel] = media[:, i]
    return data


DARK_PROBE_SPLIT = 12
"""First date carrying ``channel_1`` spend in :func:`dark_probe_funnel_mmm_data`."""


@pytest.fixture(scope="module")
def dark_probe_funnel_fitted_mmm(dark_probe_funnel_mmm_data):
    """Funnel MMM whose mediated channel is dark wherever one probe would land."""
    return _build_funnel_mmm(
        dark_probe_funnel_mmm_data, link="log", effect_cls=PartialChannelFunnelEffect
    )


@pytest.fixture(scope="module")
def funnel_time_varying_media_fitted_mmm(funnel_mmm_data):
    """Funnel MMM with a time-varying media multiplier.

    Mediation and ``time_index`` interact: the HSGP multiplier evaluates its
    basis at the window's absolute positions on the date axis, and the window is
    widened by the mediated reach, so getting either wrong moves the numbers.
    """
    return _build_funnel_mmm(funnel_mmm_data, link="log", time_varying_media=True)


@pytest.fixture(scope="module")
def two_mediators_fitted_mmm(funnel_mmm_data):
    """Funnel MMM carrying two mediated effects of different reach.

    The evaluation window has to be sized for the longer of the two, and the
    increment has to collect both contributions.  Taking the first reach, or the
    mean of them, would pass every single-effect test here.
    """
    return _build_funnel_mmm(
        funnel_mmm_data,
        link="log",
        effects=[
            _funnel_effect(prefix="funnel", demand_l_max=1),
            ChainedMediatorEffect(
                prefix="slow",
                upper_transform=_media_transformation(
                    "upper_slow", ("channel",), FUNNEL_L_MAX, beta_sigma=3.0
                ),
                demand_transform=_media_transformation(
                    "demand_slow", (), FUNNEL_L_MAX + 2, beta_sigma=3.0
                ),
            ),
        ],
    )


def _build_simple_effect_mmm(data, effect, link="log"):
    """Fit an MMM carrying one arbitrary ``mu_effect``.

    Separate from :func:`_build_funnel_mmm` because the effects it serves have
    no data variables of their own, so the model can be built from the frame the
    other identity-link fixtures use.
    """
    mmm = MMM(
        channel_columns=["channel_1", "channel_2"],
        date_column="date",
        target_column="y",
        control_columns=None,
        adstock=GeometricAdstock(l_max=FUNNEL_L_MAX),
        saturation=LogisticSaturation(),
        link=link,
    )
    mmm.add_mu_effect(effect)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mock_fit(mmm, data["X_df"], data["y_series"], random_seed=seed)
    return mmm


@pytest.fixture(scope="module")
def global_normalization_fitted_mmm(funnel_mmm_data):
    """MMM whose effect normalizes spend by its mean over the whole date axis."""
    return _build_simple_effect_mmm(
        funnel_mmm_data, GlobalNormalizationEffect(prefix="global")
    )


@pytest.fixture(scope="module")
def windowed_global_normalization_fitted_mmm(funnel_mmm_data):
    """MMM whose date-reducing effect declares ``evaluation_mode="window"``."""
    return _build_simple_effect_mmm(
        funnel_mmm_data, WindowedGlobalNormalizationEffect(prefix="global")
    )


@pytest.fixture(scope="module")
def duck_typed_baseline_effect_fitted_mmm(funnel_mmm_data):
    """MMM with a duck-typed effect that does not read spend."""
    return _build_simple_effect_mmm(
        funnel_mmm_data, DuckTypedEffect(prefix="duck", reads_spend=False)
    )


@pytest.fixture(scope="module")
def duck_typed_spend_effect_fitted_mmm(funnel_mmm_data):
    """MMM with a duck-typed effect that *does* read spend."""
    return _build_simple_effect_mmm(
        funnel_mmm_data, DuckTypedEffect(prefix="duck", reads_spend=True)
    )


@pytest.fixture(scope="module")
def negative_effect_fitted_mmm(funnel_mmm_data):
    """Additive MMM whose mediated effect subtracts from the response."""
    return _build_simple_effect_mmm(
        funnel_mmm_data, NegativeEffect(prefix="cannibal"), link="identity"
    )


@pytest.fixture(scope="module")
def panel_mediated_fitted_mmm(panel_mmm_data):
    """Panel MMM with a spend-dependent effect, so ``channel_axis`` is not zero.

    ``channel_data`` is laid out ``(date, country, channel)`` here, so a
    per-channel perturbation has to index the last axis rather than the first.
    The two happen to coincide in every non-panel fixture, which is exactly why
    this one exists.
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
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
    )
    mmm.add_mu_effect(SubsetDimsEffect(prefix="global"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mock_fit(mmm, X, panel_mmm_data["y"], random_seed=seed)
    return mmm


@pytest.fixture(scope="module")
def panel_mediated_log_link_fitted_mmm(panel_mmm_data):
    """Panel log-link MMM with a spend-dependent effect.

    Combines the three things that individually bend the layout: a panel
    ``channel_data`` laid out ``(date, country, channel)``, a mediated effect
    whose contribution drops the panel dimension, and a log link whose reducer
    broadcasts a ``(date, country)`` baseline response against a
    ``(date, country, channel)`` perturbation.

    ``LogisticSaturation`` rather than ``LogSaturation``, deliberately: the
    latter turns channel scaling off, and ``SubsetDimsEffect`` would then add
    *raw* spend to a log-scale linear predictor, overflowing ``exp(mu)``.
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
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        link="log",
    )
    mmm.add_mu_effect(SubsetDimsEffect(prefix="global"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mock_fit(mmm, X, panel_mmm_data["y"], random_seed=seed)
    return mmm


@pytest.fixture(scope="module")
def trend_effect_fitted_mmm(funnel_mmm_data):
    """MMM with a ``mu_effect`` that does not read spend.

    A linear trend cancels in a spend counterfactual, so incrementality must skip
    it without ever asking it to opt in -- otherwise every effect that already
    exists would suddenly have to.
    """
    mmm = MMM(
        channel_columns=["channel_1", "channel_2"],
        date_column="date",
        target_column="y",
        control_columns=None,
        adstock=GeometricAdstock(l_max=FUNNEL_L_MAX),
        saturation=LogisticSaturation(),
    )
    mmm.add_mu_effect(
        LinearTrendEffect(trend=LinearTrend(n_changepoints=2), prefix="trend")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mock_fit(
            mmm, funnel_mmm_data["X_df"], funnel_mmm_data["y_series"], random_seed=seed
        )
    return mmm


@pytest.fixture
def shadow_named_node():
    """Give an anonymous graph node a second claim to a name, then take it back.

    Stands in for the one hazard a stock model cannot produce on its own: a
    custom effect whose intermediate happens to be named ``mu``, which leaves
    the linear predictor's name carried by two different nodes.  The surgery is
    done on the live graph because that is exactly what the user's effect would
    do, and it is undone at teardown so that the module-scoped fitted fixtures
    stay reusable.

    Yields
    ------
    callable
        ``shadow(model, name="mu")`` names one anonymous ancestor of the
        model's observed and deterministic variables *name* and returns it.
    """
    renamed: list[tuple[Variable, str | None]] = []

    def shadow(model, name: str = "mu") -> Variable:
        pymc_model = model.model
        roots = list(pymc_model.observed_RVs) + list(pymc_model.deterministics)
        for node in ancestors(roots):
            if node.owner is not None and getattr(node, "name", None) is None:
                renamed.append((node, node.name))
                node.name = name
                return node
        raise AssertionError(f"No anonymous node available to name {name!r}.")

    yield shadow

    for node, original_name in renamed:
        node.name = original_name
