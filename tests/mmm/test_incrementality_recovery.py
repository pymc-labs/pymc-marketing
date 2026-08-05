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
"""ROAS recovery from real fits of the three ``mmm_multiplicative`` model variants.

Every other incrementality test compares the module against an oracle evaluated
on the *same* posterior, so a systematic problem upstream of the module -- the
wrong estimand, the wrong scale, a link misread at fit time -- would cancel out
of all of them.  These tests close that loop: data is simulated from the
``mmm_example`` notebook's data-generating process (library ``geometric_adstock``
and ``logistic_saturation``, additive response), the **true** ROAS is computed
from the DGP itself -- a counterfactual difference of the noise-free response,
the ``mmm_roas`` notebook's construction -- and the three model variants the
``mmm_multiplicative`` notebook fits on this same kind of data -- linear
(identity link), log, and log-log -- are each fit with real MCMC and asked to
recover it through ``mmm.incrementality.contribution_over_spend``.

The data is additive, so the linear model is well-specified and has to cover
the truth with its posterior; the log and log-log models are misspecified in
the way real models always are, and only have to land close -- which is the
``mmm_multiplicative`` notebook's own observation about the three variants.
Slow by construction (three NUTS fits), hence the marker; run with
``pytest --run-slow``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.mmm import (
    MMM,
    GeometricAdstock,
    LogisticSaturation,
    LogSaturation,
)
from pymc_marketing.mmm.transformers import geometric_adstock, logistic_saturation

SEED = sum(map(ord, "incrementality_recovery"))
N_DATES = 120
L_MAX = 8
ADSTOCK_ALPHA = {"x1": 0.4, "x2": 0.2}
SATURATION_LAM = {"x1": 4.0, "x2": 3.0}
BETA = {"x1": 3.0, "x2": 2.0}
INTERCEPT = 2.0
NOISE_SIGMA = 0.25
CHANNELS = ["x1", "x2"]

FIT_KWARGS = {
    "draws": 300,
    "tune": 300,
    "chains": 2,
    "cores": 2,
    "target_accept": 0.9,
    "random_seed": SEED,
    "progressbar": False,
}


@pytest.fixture(scope="module")
def recovery_data():
    """The ``mmm_example`` DGP, with its DGP-derived true ROAS.

    Spend patterns, transformations and coefficients follow the notebook: two
    channels (one always-on with spikes, one flighted), the library's
    ``geometric_adstock`` and ``logistic_saturation``, and an additive target.
    Trend, seasonality and events are omitted so the three fitted variants need
    no control columns.

    The true ROAS is the ``mmm_roas`` notebook's construction: the summed
    difference between the noise-free response with actual spend and with
    channel *m*'s spend zeroed over the whole range, divided by the channel's
    spend.  The DGP is additive and both transformations map zero spend to
    zero, so that difference is exactly channel *m*'s own media term.
    """
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2022-01-03", periods=N_DATES, freq="W-MON")

    x1 = rng.uniform(size=N_DATES)
    x1 = np.where(x1 > 0.9, x1, x1 / 2)
    x2 = rng.uniform(size=N_DATES)
    x2 = np.where(x2 > 0.8, x2, 0.0)
    spend = {"x1": x1, "x2": x2}

    media = {}
    for channel, values in spend.items():
        adstocked = geometric_adstock(
            x=xr.DataArray(values, dims=("date",)),
            alpha=ADSTOCK_ALPHA[channel],
            l_max=L_MAX,
            normalize=True,
            dim="date",
        )
        saturated = logistic_saturation(x=adstocked, lam=SATURATION_LAM[channel])
        media[channel] = BETA[channel] * np.asarray(saturated.eval())

    mu = INTERCEPT + media["x1"] + media["x2"]
    y = mu + rng.normal(0.0, NOISE_SIGMA, size=N_DATES)

    true_roas = {
        channel: float(media[channel].sum() / spend[channel].sum())
        for channel in CHANNELS
    }

    X = pd.DataFrame({"date": dates, "x1": x1, "x2": x2})
    return {"X": X, "y": pd.Series(y, name="y"), "true_roas": true_roas}


def _fit(recovery_data, saturation, link):
    """Fit one ``mmm_multiplicative`` variant on the shared data."""
    mmm = MMM(
        date_column="date",
        channel_columns=CHANNELS,
        adstock=GeometricAdstock(l_max=L_MAX),
        saturation=saturation,
        link=link,
    )
    with warnings.catch_warnings():
        # link="log" is flagged experimental and LogSaturation overrides the
        # channel scaling; both are the point of the fixture.
        warnings.simplefilter("ignore", UserWarning)
        mmm.fit(recovery_data["X"], recovery_data["y"], **FIT_KWARGS)
    return mmm


@pytest.fixture(scope="module")
def linear_recovery_mmm(recovery_data):
    """The notebook's additive variant: identity link, logistic saturation."""
    return _fit(recovery_data, LogisticSaturation(), "identity")


@pytest.fixture(scope="module")
def log_recovery_mmm(recovery_data):
    """The notebook's multiplicative variant: log link, logistic saturation."""
    return _fit(recovery_data, LogisticSaturation(), "log")


@pytest.fixture(scope="module")
def loglog_recovery_mmm(recovery_data):
    """The notebook's log-log variant: log link, ``LogSaturation`` on raw spend."""
    return _fit(recovery_data, LogSaturation(), "log")


def _posterior_roas(mmm):
    """All-time ROAS samples per channel, flattened over (chain, draw)."""
    roas = mmm.incrementality.contribution_over_spend(frequency="all_time")
    return roas.stack(sample=("chain", "draw"))


@pytest.mark.slow
class TestRoasRecovery:
    """The three notebook variants recover the DGP's ROAS, and agree."""

    # The misspecified models absorb the DGP's additive structure into their
    # own multiplicative form, so bias rather than posterior width dominates;
    # their bound is on accuracy, not coverage.
    MISSPECIFIED_RTOL = 0.35
    WELL_SPECIFIED_RTOL = 0.15

    @pytest.mark.parametrize(
        "model_fixture, rtol",
        [
            ("linear_recovery_mmm", WELL_SPECIFIED_RTOL),
            ("log_recovery_mmm", MISSPECIFIED_RTOL),
            ("loglog_recovery_mmm", MISSPECIFIED_RTOL),
        ],
    )
    def test_posterior_median_roas_is_close_to_the_truth(
        self, request, recovery_data, model_fixture, rtol
    ):
        """Each variant's posterior-median ROAS lands within tolerance per channel."""
        mmm = request.getfixturevalue(model_fixture)
        roas = _posterior_roas(mmm).median("sample")

        for channel, truth in recovery_data["true_roas"].items():
            estimate = float(roas.sel(channel=channel))
            assert abs(estimate - truth) < rtol * truth, (
                f"{model_fixture}: channel {channel} ROAS {estimate:.4f} vs "
                f"true {truth:.4f}"
            )

    def test_the_well_specified_model_covers_the_truth(
        self, recovery_data, linear_recovery_mmm
    ):
        """The linear model has the DGP's form, so its posterior has to reach the truth.

        The 99% interval rather than the conventional 94%, with a small relative
        slack: the model is well-specified in functional form, but the default
        priors shrink the spiky channel's coefficient slightly, and with a tight
        posterior that bias parks the truth at the interval's edge.  What the
        test rules out is the truth sitting *clear* of the posterior -- the
        signature of a wrong estimand rather than of shrinkage.
        """
        samples = _posterior_roas(linear_recovery_mmm)
        slack = 0.02

        for channel, truth in recovery_data["true_roas"].items():
            channel_samples = samples.sel(channel=channel)
            low = float(channel_samples.quantile(0.005))
            high = float(channel_samples.quantile(0.995))
            assert low * (1 - slack) < truth < high * (1 + slack), (
                f"channel {channel}: true ROAS {truth:.4f} outside [{low:.4f}, "
                f"{high:.4f}]"
            )

    def test_the_three_variants_agree_with_each_other(
        self, linear_recovery_mmm, log_recovery_mmm, loglog_recovery_mmm
    ):
        """The estimand is a property of the data, not of the link chosen.

        All three models see the same spend and the same response, so their
        ROAS estimates have to be close to *each other* as well as to the truth
        -- this is the claim the ``mmm_multiplicative`` notebook makes when it
        compares the variants' channel contributions.
        """
        medians = {
            name: _posterior_roas(mmm).median("sample")
            for name, mmm in {
                "linear": linear_recovery_mmm,
                "log": log_recovery_mmm,
                "loglog": loglog_recovery_mmm,
            }.items()
        }

        for channel in CHANNELS:
            estimates = np.array(
                [float(roas.sel(channel=channel)) for roas in medians.values()]
            )
            spread = estimates.max() / estimates.min() - 1.0
            assert spread < 0.35, f"channel {channel}: estimates {estimates}"

    def test_the_pre_fix_computation_would_not_recover_the_truth(
        self, recovery_data, log_recovery_mmm
    ):
        """The linear-predictor version this PR replaced fails this suite.

        Before the fix, incrementality summed ``channel_contribution``
        differences and multiplied by ``target_scale`` -- under a log link that
        is a log-space quantity wearing response-scale units.  The saturation
        maps zero spend to zero contribution, so the pre-fix all-time number is
        simply ``channel_contribution.sum("date") * target_scale``, and on this
        data it misses the truth by 60-70% per channel while the fixed
        estimator lands within 10%.  End-to-end negative control: an accidental
        revert cannot pass.

        Only the log variant serves as the control: for the log-log variant on
        this DGP the pre-fix number for the spiky channel happens to land
        within the model-misspecification range (~28%), so it could not
        discriminate a revert from an imperfect model.
        """
        mmm = log_recovery_mmm
        posterior = mmm.idata.posterior

        pre_fix = (
            posterior["channel_contribution"].sum("date") * mmm.data.get_target_scale()
        ).median(("chain", "draw"))
        spend = mmm.data.get_channel_spend().sum("date")
        pre_fix_roas = pre_fix / spend

        for channel, truth in recovery_data["true_roas"].items():
            estimate = float(pre_fix_roas.sel(channel=channel))
            assert abs(estimate - truth) > 0.5 * truth, (
                f"channel {channel}: pre-fix ROAS {estimate:.4f} is "
                f"unexpectedly close to the truth {truth:.4f}"
            )
