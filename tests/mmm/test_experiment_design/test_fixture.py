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

"""Tests for fixture generation and from_idata round-trip."""

import numpy as np
import pytest
import xarray as xr

from pymc_marketing.mmm.experiment_design import (
    ExperimentDesigner,
    generate_experiment_fixture,
)
from pymc_marketing.mmm.experiment_design.fixture import _simulate_spend


@pytest.fixture(scope="module")
def synthetic_idata():
    """Generate a synthetic (fast) fixture for testing."""
    return generate_experiment_fixture(
        channels=["tv", "search", "social"],
        fit_model=False,
        seed=42,
    )


class TestGenerateExperimentFixture:
    """Tests for generate_experiment_fixture."""

    def test_returns_inference_data(self, synthetic_idata):
        assert isinstance(synthetic_idata, xr.DataTree)

    def test_has_posterior(self, synthetic_idata):
        assert hasattr(synthetic_idata, "posterior")

    def test_has_constant_data(self, synthetic_idata):
        assert hasattr(synthetic_idata, "constant_data")

    def test_noise_matches_injected_residuals_on_scaled_target(self, synthetic_idata):
        constant = synthetic_idata.constant_data
        n_weeks = constant.attrs["n_weeks"]
        rng = np.random.default_rng(42)
        rng.standard_normal((n_weeks, 3))
        residuals = rng.normal(0, constant.attrs["noise_std"], n_weeks)
        assert constant.residual_std.item() == pytest.approx(
            residuals.std() / constant.target_scale.item()
        )

    def test_synthetic_posterior_uses_scaled_channel_and_target_parameters(
        self, synthetic_idata
    ):
        cd = synthetic_idata.constant_data
        posterior = synthetic_idata.posterior
        lam = posterior.saturation_lam.sel(channel="tv").mean().item()
        beta = posterior.saturation_beta.sel(channel="tv").mean().item()
        assert lam / cd.channel_scale.sel(channel="tv").item() == pytest.approx(
            0.5, rel=0.02
        )
        assert beta * cd.target_scale.item() == pytest.approx(3.0, rel=0.02)

    def test_fitted_fixture_keeps_training_data_and_uses_known_noise(self):
        idata = generate_experiment_fixture(fit_model=True, n_weeks=30, l_max=3)
        cd = idata.constant_data
        rng = np.random.default_rng(42)
        rng.standard_normal((30, 3))
        noise = rng.normal(0, cd.attrs["noise_std"], 30)
        assert cd.residual_std.item() * cd.target_scale.item() == pytest.approx(
            noise.std()
        )
        expected_spend = (
            cd.channel_data.isel(date=slice(-8, None)) / cd.channel_scale
        ).mean("date")
        np.testing.assert_allclose(cd.current_weekly_spend, expected_spend)
        designer = ExperimentDesigner.from_idata(idata)
        assert (
            len(designer.recommend(spend_changes=[0.2], durations=[4], min_snr=0)) == 3
        )

    @pytest.mark.parametrize(
        "options", [{"saturation": "hill"}, {"adstock": "weibull"}]
    )
    def test_rejects_mislabeled_fixture(self, options):
        with pytest.raises(NotImplementedError, match="unsupported"):
            generate_experiment_fixture(fit_model=False, **options)

    def test_real_fit_does_not_silently_ignore_synthetic_counts(self):
        with pytest.raises(ValueError, match="synthetic draws"):
            generate_experiment_fixture(fit_model=True, n_draws=12)

    def test_posterior_has_expected_variables(self, synthetic_idata):
        posterior = synthetic_idata.posterior
        assert "saturation_lam" in posterior
        assert "saturation_beta" in posterior
        assert "adstock_alpha" in posterior

    def test_posterior_channel_dim(self, synthetic_idata):
        posterior = synthetic_idata.posterior
        assert "channel" in posterior["saturation_lam"].dims

    def test_constant_data_has_metadata(self, synthetic_idata):
        cd = synthetic_idata.constant_data
        assert "current_weekly_spend" in cd
        assert "residual_std" in cd
        assert "l_max" in cd
        assert "normalize" in cd

    def test_spend_correlation_present(self, synthetic_idata):
        assert "spend_correlation" in synthetic_idata.constant_data

    def test_channel_coords_match(self, synthetic_idata):
        channels = list(
            synthetic_idata.posterior["saturation_lam"].coords["channel"].values
        )
        assert channels == ["tv", "search", "social"]

    def test_parameters_positive(self, synthetic_idata):
        posterior = synthetic_idata.posterior
        assert (posterior["saturation_lam"].values > 0).all()
        assert (posterior["saturation_beta"].values > 0).all()
        assert (posterior["adstock_alpha"].values > 0).all()
        assert (posterior["adstock_alpha"].values < 1).all()


class TestFromIdataRoundTrip:
    """Tests for ExperimentDesigner.from_idata."""

    def test_creates_designer(self, synthetic_idata):
        designer = ExperimentDesigner.from_idata(synthetic_idata)
        assert isinstance(designer, ExperimentDesigner)

    def test_channels_recovered(self, synthetic_idata):
        designer = ExperimentDesigner.from_idata(synthetic_idata)
        assert designer.channel_columns == ["tv", "search", "social"]

    def test_channels_recovered_with_scalar_first_in_posterior(self, synthetic_idata):
        idata = synthetic_idata.copy()
        posterior = idata.posterior.dataset
        idata["posterior"] = posterior.assign(intercept=0)[
            ["intercept", *posterior.data_vars]
        ]

        designer = ExperimentDesigner.from_idata(idata)

        assert designer.channel_columns == ["tv", "search", "social"]

    def test_l_max_recovered(self, synthetic_idata):
        designer = ExperimentDesigner.from_idata(synthetic_idata)
        assert designer.l_max == 8

    def test_normalize_recovered(self, synthetic_idata):
        designer = ExperimentDesigner.from_idata(synthetic_idata)
        assert designer.normalize is True

    def test_posterior_has_correct_draws(self, synthetic_idata):
        designer = ExperimentDesigner.from_idata(synthetic_idata)
        expected_draws = 2 * 2000  # n_chains * n_draws
        assert designer.n_draws == expected_draws

    def test_netcdf_round_trip(self, synthetic_idata, tmp_path):
        path = tmp_path / "test_fixture.nc"
        synthetic_idata.to_netcdf(path)
        with xr.open_datatree(path) as loaded:
            designer = ExperimentDesigner.from_idata(loaded)
        assert designer.channel_columns == ["tv", "search", "social"]
        assert designer.n_draws == 2 * 2000

    def test_unsupported_saturation_raises(self, synthetic_idata):
        with pytest.raises(NotImplementedError, match="Saturation"):
            ExperimentDesigner.from_idata(synthetic_idata, saturation="hill")

    def test_unsupported_adstock_raises(self, synthetic_idata):
        with pytest.raises(NotImplementedError, match="Adstock"):
            ExperimentDesigner.from_idata(synthetic_idata, adstock="weibull")


class TestSimulateSpend:
    """Tests for _simulate_spend helper."""

    def test_default_rng(self):
        """Calling without rng uses the default generator (covers line 60)."""
        spend = _simulate_spend(n_weeks=10, n_channels=2)
        assert spend.shape == (10, 2)
        assert np.all(spend > 0)


class TestFixtureShortResiduals:
    """Test fixture with very short time series."""

    def test_short_series_autocorr_zero(self):
        """With n_weeks=2, autocorrelation defaults to 0.0 (covers line 211)."""
        idata = generate_experiment_fixture(
            channels=["ch1"],
            true_params={"ch1": {"lam": 1.0, "beta": 1.0, "alpha": 0.5}},
            n_weeks=2,
            fit_model=False,
            seed=1,
        )
        autocorr = float(idata.constant_data["residual_autocorr"].values)
        assert autocorr == 0.0
