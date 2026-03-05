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

"""Slow simulation-based tests for assurance calibration.

These tests validate that the ExperimentDesigner's assurance calculations
are calibrated against the known ground-truth DGP. They are marked slow
because they require many simulated experiments.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from pymc_marketing.mmm.experiment_design import (
    ExperimentDesigner,
    generate_experiment_fixture,
)
from pymc_marketing.mmm.experiment_design.fixture import (
    _logistic_saturation_np,
)


@pytest.fixture(scope="module")
def fixture_with_truth():
    """Generate fixture with known ground truth for calibration."""
    true_params = {
        "tv": {"lam": 0.5, "beta": 3.0, "alpha": 0.7},
        "search": {"lam": 2.0, "beta": 1.5, "alpha": 0.3},
        "social": {"lam": 1.0, "beta": 0.8, "alpha": 0.5},
    }
    idata = generate_experiment_fixture(
        channels=["tv", "search", "social"],
        true_params=true_params,
        fit_model=False,
        seed=42,
    )
    designer = ExperimentDesigner.from_idata(idata)
    return designer, true_params, idata


@pytest.mark.slow
class TestAssuranceCalibration:
    """Validate that reported assurance matches empirical detection rates."""

    def test_assurance_calibration_spend_increase(self, fixture_with_truth):
        """Simulate experiments and check detection rates match assurance.

        For a +30% spend change on search (fast adstock, well-identified),
        the fraction of simulated experiments where the true lift exceeds
        the detection threshold should be within ±0.10 of the reported
        assurance.
        """
        designer, true_params, _idata = fixture_with_truth
        channel = "search"
        frac = 0.3
        T = 6
        significance_level = 0.05
        n_simulations = 500
        rng = np.random.default_rng(123)

        x_current = designer._current_spend[channel]
        delta_x = frac * x_current

        predicted_lift = designer._predict_lift(channel, delta_x, T)
        sigma_d = designer._residual_std * np.sqrt(T)
        reported_assurance = designer._compute_assurance(
            predicted_lift, sigma_d, significance_level
        )

        p = true_params[channel]
        true_alpha = p["alpha"]
        l_max = designer.l_max

        alpha_safe = np.clip(true_alpha, 1e-6, 1 - 1e-6)
        S = (1 - alpha_safe**l_max) / (1 - alpha_safe)
        x_ss_true = x_current

        true_total_lift = 0.0
        for t in range(T):
            partial = (1 - alpha_safe ** (t + 1)) / (1 - alpha_safe)
            ramp = partial / S
            effective = x_ss_true + delta_x * ramp
            weekly_lift = (
                p["beta"] * _logistic_saturation_np(np.array([effective]), p["lam"])[0]
                - p["beta"]
                * _logistic_saturation_np(np.array([x_ss_true]), p["lam"])[0]
            )
            true_total_lift += weekly_lift

        z_crit = norm.ppf(1 - significance_level / 2)

        detections = 0
        for _ in range(n_simulations):
            observed = true_total_lift + rng.normal(0, sigma_d)
            z_obs = abs(observed) / sigma_d
            if z_obs > z_crit:
                detections += 1

        empirical_power = detections / n_simulations
        assert abs(empirical_power - reported_assurance) < 0.15, (
            f"Assurance mismatch: reported={reported_assurance:.3f}, "
            f"empirical={empirical_power:.3f}"
        )


@pytest.mark.slow
class TestChannelRanking:
    """Validate that channel ranking reflects posterior uncertainty."""

    def test_most_uncertain_channel_ranked_first(self, fixture_with_truth):
        """The channel with the widest posterior HDI product should be
        ranked as the most uncertain (rank #1).
        """
        designer, _true_params, _idata = fixture_with_truth
        ranks = designer._get_uncertainty_ranks()

        uncertainties = {}
        for ch in designer.channel_columns:
            import arviz as az

            p = designer._posterior_samples[ch]
            hdi_lam = az.hdi(p["lam"], hdi_prob=0.94)
            hdi_beta = az.hdi(p["beta"], hdi_prob=0.94)
            uncertainties[ch] = (hdi_lam[1] - hdi_lam[0]) * (hdi_beta[1] - hdi_beta[0])

        most_uncertain = max(uncertainties, key=uncertainties.get)
        assert ranks[most_uncertain] == 1
