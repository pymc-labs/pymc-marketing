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
"""Tests for the link module and multiplicative MMM functionality.

Covers the phased test plan:
    Layer 1: API validation (link parameter, LinkSpec, guardrails)
    Layer 2: LogSaturation component
    Layer 3: build_model deterministics
    Layer 4: Decomposition consistency
    Layer 5: Budget optimizer under log link
    Layer 6: Serialization and backward compatibility
"""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from pymc_extras.prior import Prior

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation, LogSaturation
from pymc_marketing.mmm.link import (
    IdentityLinkSpec,
    LinkFunction,
    LinkSpec,
    LogLinkSpec,
    get_link_spec,
)
from pymc_marketing.mmm.multidimensional import MMM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_positive_panel(
    n_dates: int = 8,
    countries: tuple[str, ...] = ("A", "B"),
    channels: tuple[str, ...] = ("C1", "C2"),
    target_column: str = "y",
) -> tuple[pd.DataFrame, pd.Series]:
    """Synthetic panel with strictly positive target values."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-06", periods=n_dates, freq="W-MON")
    rows = []
    for d in dates:
        for c in countries:
            row = {"date": d, "country": c}
            for ch in channels:
                row[ch] = rng.uniform(10, 100)
            row[target_column] = rng.uniform(50, 500)
            rows.append(row)
    df = pd.DataFrame(rows)
    y = df.pop(target_column)
    y.name = target_column
    return df, y


def _make_mmm(link: str = "identity", dims=("country",), **kwargs) -> MMM:
    sat = LogSaturation() if link == "log" else LogisticSaturation()
    return MMM(
        date_column="date",
        channel_columns=["C1", "C2"],
        adstock=GeometricAdstock(l_max=4),
        saturation=sat,
        dims=dims,
        link=link,
        **kwargs,
    )


# ===========================================================================
# Layer 1: API validation (unit tests, no fitting)
# ===========================================================================
class TestLinkAPI:
    """Test LinkFunction enum, LinkSpec, and MMM parameter validation."""

    def test_link_default_is_identity(self):
        mmm = MMM(
            date_column="date",
            channel_columns=["C1"],
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
        )
        assert mmm.link == LinkFunction.IDENTITY

    @pytest.mark.parametrize("link_val", ["identity", "log"])
    def test_link_accepts_string(self, link_val):
        mmm = _make_mmm(link=link_val, dims=None)
        assert mmm.link == LinkFunction(link_val)

    def test_link_invalid_value_raises(self):
        with pytest.raises((ValueError, ValidationError)):
            _make_mmm(link="sqrt", dims=None)

    def test_log_link_default_likelihood_is_lognormal(self):
        mmm = _make_mmm(link="log", dims=None)
        assert mmm.model_config["likelihood"].distribution == "LogNormal"

    def test_identity_link_default_likelihood_is_normal(self):
        mmm = _make_mmm(link="identity", dims=None)
        assert mmm.model_config["likelihood"].distribution == "Normal"

    def test_link_likelihood_incompatible_raises(self, mock_pymc_sample):
        mmm = _make_mmm(
            link="log",
            dims=None,
            model_config={
                "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=0.5)),
            },
        )
        X, y = _make_positive_panel(n_dates=4, countries=("A",))
        X = X.drop(columns=["country"])
        with pytest.raises(ValueError, match="not compatible with link"):
            mmm.build_model(X, y)

    def test_log_link_negative_target_raises(self, mock_pymc_sample):
        mmm = _make_mmm(link="log", dims=None)
        X, y = _make_positive_panel(n_dates=4, countries=("A",))
        X = X.drop(columns=["country"])
        y.iloc[0] = -1.0
        with pytest.raises(ValueError, match="strictly positive"):
            mmm.build_model(X, y)

    def test_log_link_zero_target_raises(self, mock_pymc_sample):
        mmm = _make_mmm(link="log", dims=None)
        X, y = _make_positive_panel(n_dates=4, countries=("A",))
        X = X.drop(columns=["country"])
        y.iloc[0] = 0.0
        with pytest.raises(ValueError, match="strictly positive"):
            mmm.build_model(X, y)

    def test_log_link_mu_effects_warning(self, mock_pymc_sample):
        from pymc_marketing.mmm.additive_effect import LinearTrendEffect
        from pymc_marketing.mmm.linear_trend import LinearTrend

        mmm = _make_mmm(link="log", dims=None)
        mmm.mu_effects.append(
            LinearTrendEffect(
                trend=LinearTrend(),
                prefix="trend",
                date_dim_name="date",
            )
        )
        X, y = _make_positive_panel(n_dates=4, countries=("A",))
        X = X.drop(columns=["country"])
        with pytest.warns(UserWarning, match="mu_effects"):
            mmm.build_model(X, y)


# ===========================================================================
# Layer 1b: LinkSpec unit tests
# ===========================================================================
class TestLinkSpec:
    """Direct tests for the LinkSpec abstraction."""

    def test_get_link_spec_identity(self):
        spec = get_link_spec(LinkFunction.IDENTITY)
        assert isinstance(spec, IdentityLinkSpec)

    def test_get_link_spec_log(self):
        spec = get_link_spec(LinkFunction.LOG)
        assert isinstance(spec, LogLinkSpec)

    def test_identity_validate_target_accepts_negatives(self):
        spec = IdentityLinkSpec()
        spec.validate_target(np.array([-1.0, 0.0, 1.0]))

    def test_log_validate_target_rejects_negatives(self):
        spec = LogLinkSpec()
        with pytest.raises(ValueError):
            spec.validate_target(np.array([-1.0, 1.0]))

    def test_log_validate_target_rejects_zeros(self):
        spec = LogLinkSpec()
        with pytest.raises(ValueError):
            spec.validate_target(np.array([0.0, 1.0]))

    def test_log_validate_target_accepts_positive(self):
        spec = LogLinkSpec()
        spec.validate_target(np.array([0.1, 1.0, 100.0]))

    @pytest.mark.parametrize(
        "likelihood",
        [
            Prior("Normal", sigma=1),
            Prior("StudentT", nu=3, sigma=1),
            Prior("LogNormal", sigma=1),
            Prior("TruncatedNormal", sigma=1, lower=0),
        ],
    )
    def test_validate_likelihood_compat_identity_accepts_any(self, likelihood):
        LinkSpec.validate_likelihood_compatibility(LinkFunction.IDENTITY, likelihood)

    def test_validate_likelihood_compat_log_lognormal(self):
        LinkSpec.validate_likelihood_compatibility(
            LinkFunction.LOG, Prior("LogNormal", sigma=1)
        )

    def test_validate_likelihood_compat_log_normal_raises(self):
        with pytest.raises(ValueError, match="not compatible"):
            LinkSpec.validate_likelihood_compatibility(
                LinkFunction.LOG, Prior("Normal", sigma=1)
            )


# ===========================================================================
# Layer 2: LogSaturation component tests
# ===========================================================================
class TestLogSaturation:
    """Targeted tests for LogSaturation beyond the auto-discovered parametrized suite."""

    def test_log_saturation_at_zero(self):
        sat = LogSaturation()
        prior = sat.sample_prior(random_seed=np.random.default_rng(0))
        curve = sat.sample_curve(prior)
        assert np.allclose(curve.sel(x=0.0).values, 0.0, atol=1e-7)

    def test_log_saturation_monotonic(self):
        sat = LogSaturation()
        prior = sat.sample_prior(random_seed=np.random.default_rng(0))
        curve = sat.sample_curve(prior, max_value=10.0, num_points=200)
        mean_curve = curve.mean(dim=("chain", "draw"))
        diffs = np.diff(mean_curve.values)
        assert np.all(diffs >= 0), (
            "LogSaturation should be monotonically non-decreasing"
        )

    def test_log_saturation_serialization_round_trip(self):
        sat = LogSaturation()
        d = sat.to_dict()
        from pymc_marketing.mmm.components.saturation import saturation_from_dict

        sat2 = saturation_from_dict(d)
        assert isinstance(sat2, LogSaturation)
        assert sat2.to_dict() == d


# ===========================================================================
# Layer 3: build_model deterministics (integration, build only)
# ===========================================================================
class TestBuildModelDeterministics:
    """Test that build_model creates the correct deterministic variables."""

    @pytest.mark.parametrize("link", ["identity", "log"])
    def test_build_model_has_total_media(self, link, mock_pymc_sample):
        mmm = _make_mmm(link=link)
        X, y = _make_positive_panel()
        mmm.build_model(X, y)
        assert "total_media_contribution_original_scale" in mmm.model.named_vars

    def test_build_model_identity_no_y_original_scale(self, mock_pymc_sample):
        mmm = _make_mmm(link="identity")
        X, y = _make_positive_panel()
        mmm.build_model(X, y)
        assert "y_original_scale" not in mmm.model.named_vars

    def test_build_model_log_has_y_original_scale(self, mock_pymc_sample):
        mmm = _make_mmm(link="log")
        X, y = _make_positive_panel()
        mmm.build_model(X, y)
        assert "y_original_scale" in mmm.model.named_vars

    def test_build_model_log_y_original_scale_uses_output_var(self, mock_pymc_sample):
        mmm = _make_mmm(link="log")
        X, y = _make_positive_panel()
        mmm.build_model(X, y)
        expected_name = f"{mmm.output_var}_original_scale"
        assert expected_name in mmm.model.named_vars

    @pytest.mark.parametrize("link", ["identity", "log"])
    def test_build_model_has_channel_contribution(self, link, mock_pymc_sample):
        mmm = _make_mmm(link=link)
        X, y = _make_positive_panel()
        mmm.build_model(X, y)
        assert "channel_contribution" in mmm.model.named_vars

    def test_build_model_log_add_original_scale_raises(self, mock_pymc_sample):
        mmm = _make_mmm(link="log")
        X, y = _make_positive_panel()
        mmm.build_model(X, y)
        with pytest.raises(ValueError, match="not supported for log-link"):
            mmm.add_original_scale_contribution_variable(["channel_contribution"])


# ===========================================================================
# Layer 4: Decomposition consistency (requires fit)
# ===========================================================================
class TestDecomposition:
    """Test that decomposition output is self-consistent for both link types."""

    def test_identity_decomposition_returns_dataframe(self, mock_pymc_sample):
        mmm = _make_mmm(link="identity")
        X, y = _make_positive_panel()
        mmm.fit(X, y)
        df = mmm.compute_mean_contributions_over_time()
        assert isinstance(df, pd.DataFrame)
        assert "C1" in df.columns
        assert "C2" in df.columns
        assert "intercept" in df.columns

    def test_log_decomposition_returns_dataframe(self, mock_pymc_sample):
        mmm = _make_mmm(link="log")
        X, y = _make_positive_panel()
        mmm.fit(X, y)
        df = mmm.compute_mean_contributions_over_time()
        assert isinstance(df, pd.DataFrame)
        assert "C1" in df.columns
        assert "C2" in df.columns
        assert "intercept" in df.columns

    def test_log_decomposition_channels_non_negative(self, mock_pymc_sample):
        mmm = _make_mmm(link="log")
        X, y = _make_positive_panel()
        mmm.fit(X, y)
        df = mmm.compute_mean_contributions_over_time()
        for ch in ["C1", "C2"]:
            assert (df[ch] >= -1e-6).all(), f"Channel {ch} has negative contributions"

    def test_both_links_same_columns(self, mock_pymc_sample):
        identity_mmm = _make_mmm(link="identity")
        X, y = _make_positive_panel()
        identity_mmm.fit(X, y)
        identity_df = identity_mmm.compute_mean_contributions_over_time()

        log_mmm = _make_mmm(link="log")
        log_mmm.fit(X, y)
        log_df = log_mmm.compute_mean_contributions_over_time()

        assert set(identity_df.columns) == set(log_df.columns)


# ===========================================================================
# Layer 5: __eq__ and config consistency
# ===========================================================================
class TestEquality:
    """Test that link is included in equality comparison."""

    def test_different_links_not_equal(self):
        mmm_id = _make_mmm(link="identity", dims=None)
        mmm_log = _make_mmm(link="log", dims=None)
        assert mmm_id != mmm_log

    def test_same_link_equal(self):
        mmm1 = _make_mmm(link="log", dims=None)
        mmm2 = _make_mmm(link="log", dims=None)
        assert mmm1 == mmm2


# ===========================================================================
# Layer 6: Serialization and backward compatibility
# ===========================================================================
class TestSerialization:
    """Test save/load round-trip for link parameter."""

    def test_save_load_identity(self, mock_pymc_sample, tmp_path):
        mmm = _make_mmm(link="identity")
        X, y = _make_positive_panel()
        mmm.fit(X, y)

        path = tmp_path / "mmm_identity.nc"
        mmm.save(str(path))
        loaded = MMM.load(str(path))
        assert loaded.link == LinkFunction.IDENTITY
        assert loaded == mmm

    def test_save_load_log(self, mock_pymc_sample, tmp_path):
        mmm = _make_mmm(link="log")
        X, y = _make_positive_panel()
        mmm.fit(X, y)

        path = tmp_path / "mmm_log.nc"
        mmm.save(str(path))
        loaded = MMM.load(str(path))
        assert loaded.link == LinkFunction.LOG
        assert loaded == mmm

    def test_idata_attrs_contain_link(self, mock_pymc_sample):
        mmm = _make_mmm(link="log")
        X, y = _make_positive_panel()
        mmm.fit(X, y)
        attrs = mmm.create_idata_attrs()
        assert attrs["link"] == "log"

    def test_attrs_to_init_kwargs_defaults_identity(self):
        attrs = {
            "model_config": "{}",
            "date_column": "date",
            "control_columns": "null",
            "channel_columns": '["C1"]',
            "adstock": '{"lookup_name": "geometric", "l_max": 4}',
            "saturation": '{"lookup_name": "logistic"}',
            "adstock_first": "true",
            "yearly_seasonality": "null",
            "time_varying_intercept": "false",
            "target_column": "y",
            "time_varying_media": "false",
            "sampler_config": "{}",
            "dims": "[]",
        }
        kwargs = MMM.attrs_to_init_kwargs(attrs)
        assert kwargs["link"] == "identity"
