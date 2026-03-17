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

import arviz as az
import numpy as np
import pytest

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
        assert isinstance(synthetic_idata, az.InferenceData)

    def test_has_posterior(self, synthetic_idata):
        assert hasattr(synthetic_idata, "posterior")

    def test_has_constant_data(self, synthetic_idata):
        assert hasattr(synthetic_idata, "constant_data")

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

    def test_l_max_recovered(self, synthetic_idata):
        designer = ExperimentDesigner.from_idata(synthetic_idata)
        assert designer.l_max == 8

    def test_normalize_recovered(self, synthetic_idata):
        designer = ExperimentDesigner.from_idata(synthetic_idata)
        assert designer.normalize is True

    def test_posterior_samples_have_correct_shape(self, synthetic_idata):
        designer = ExperimentDesigner.from_idata(synthetic_idata)
        expected_draws = 2 * 2000  # n_chains * n_draws
        for ch in designer.channel_columns:
            assert len(designer._posterior_samples[ch]["lam"]) == expected_draws
            assert len(designer._posterior_samples[ch]["beta"]) == expected_draws
            assert len(designer._posterior_samples[ch]["alpha"]) == expected_draws

    def test_netcdf_round_trip(self, synthetic_idata, tmp_path):
        path = tmp_path / "test_fixture.nc"
        az.to_netcdf(synthetic_idata, str(path))
        loaded = az.from_netcdf(str(path))

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
        spend = _simulate_spend(n_weeks=10, n_channels=2, channel_names=["a", "b"])
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
