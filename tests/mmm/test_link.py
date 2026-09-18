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

import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytest
import xarray as xr
from pydantic import ValidationError
from pymc_extras.prior import Censored, Prior, Scaled
from scipy import stats

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation, LogSaturation
from pymc_marketing.mmm.additive_effect import MuEffect
from pymc_marketing.mmm.link import (
    IdentityLinkSpec,
    LinkFunction,
    LinkSpec,
    LogLinkSpec,
    _distribution_name,
    get_link_spec,
)
from pymc_marketing.mmm.mmm import MMM, BudgetOptimizerWrapper
from pymc_marketing.mmm.scaling import DataDerivedScaling, FixedScaling, Scaling
from pymc_marketing.serialization import serialization
from pymc_marketing.special_priors import LogNormalPrior


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


def _make_panel_with_target(y_values, channels=("C1", "C2")):
    """Single-dim panel whose target is exactly *y_values*."""
    rng = np.random.default_rng(0)
    n = len(y_values)
    df = pd.DataFrame({"date": pd.date_range("2025-01-06", periods=n, freq="W-MON")})
    for ch in channels:
        df[ch] = rng.uniform(10, 100, n)
    return df, pd.Series(np.asarray(y_values, dtype=float), name="y")


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

    def test_log_link_emits_experimental_warning(self):
        with pytest.warns(UserWarning, match="experimental"):
            _make_mmm(link="log", dims=None)

    def test_identity_link_does_not_emit_experimental_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _make_mmm(link="identity", dims=None)
        assert not any("experimental" in str(w.message) for w in caught)

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
            Prior("TruncatedNormal", sigma=1, lower=0),
            Prior("Gamma", sigma=1),
            Prior("Laplace", b=1),
            Prior("InverseGamma", sigma=1),
        ],
    )
    def test_validate_likelihood_compat_identity_accepts_response_scale(
        self, likelihood
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            LinkSpec.validate_likelihood_compatibility(
                LinkFunction.IDENTITY, likelihood
            )

    def test_validate_likelihood_compat_identity_lognormal_raises(self):
        with pytest.raises(ValueError) as excinfo:
            LinkSpec.validate_likelihood_compatibility(
                LinkFunction.IDENTITY, Prior("LogNormal", sigma=1)
            )
        message = str(excinfo.value)
        assert "not compatible with link='identity'" in message
        # The message has to carry the recipe for repairing a saved model,
        # since the check also runs on the load path.
        assert "idata_to_init_kwargs" in message

    def test_identity_error_names_lognormal_prior_as_class_not_distribution(self):
        # 'LogNormalPrior' is a special_priors class admitted by class name,
        # not a distribution accepted by Prior(...); the message must not
        # present it inside the distribution-name list.
        with pytest.raises(ValueError) as excinfo:
            LinkSpec.validate_likelihood_compatibility(
                LinkFunction.IDENTITY, Prior("LogNormal", sigma=1)
            )
        message = str(excinfo.value)
        assert "pymc_marketing.special_priors.LogNormalPrior" in message
        assert "'LogNormalPrior'" not in message

    def test_identity_warning_names_lognormal_prior_as_class_not_distribution(self):
        with pytest.warns(UserWarning) as records:
            LinkSpec.validate_likelihood_compatibility(
                LinkFunction.IDENTITY, Prior("Weibull", alpha=1, beta=1)
            )
        message = str(records[0].message)
        assert "pymc_marketing.special_priors.LogNormalPrior" in message
        assert "'LogNormalPrior'" not in message

    def test_validate_likelihood_compat_identity_looks_through_censored(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            LinkSpec.validate_likelihood_compatibility(
                LinkFunction.IDENTITY, Censored(Prior("Normal", sigma=1), lower=0)
            )

    def test_validate_likelihood_compat_log_looks_through_censored(self):
        LinkSpec.validate_likelihood_compatibility(
            LinkFunction.LOG, Censored(Prior("LogNormal", sigma=1), lower=0)
        )

    def test_validate_likelihood_compat_identity_unknown_warns(self):
        with pytest.warns(UserWarning, match="not a known response-scale likelihood"):
            LinkSpec.validate_likelihood_compatibility(
                LinkFunction.IDENTITY, Prior("Weibull", alpha=1, beta=1)
            )

    def test_validate_likelihood_compat_identity_unnamed_uses_class_name(self):
        class Unnamed:
            pass

        with pytest.warns(UserWarning, match="'Unnamed' is not a known"):
            LinkSpec.validate_likelihood_compatibility(LinkFunction.IDENTITY, Unnamed())

    def test_distribution_name_falls_back_to_class_name(self):
        # The name drives the set lookups, so an object without a
        # ``distribution`` has to resolve to something other than None.
        class Unnamed:
            pass

        assert _distribution_name(Prior("Normal", sigma=1)) == "Normal"
        assert (
            _distribution_name(Censored(Prior("Normal", sigma=1), lower=0)) == "Normal"
        )
        assert _distribution_name(Unnamed()) == "Unnamed"

    def test_validate_likelihood_compat_identity_accepts_lognormal_prior(self):
        # LogNormalPrior's mean is the response-scale expectation, so it is
        # admitted under identity via its class name (#2858).
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            LinkSpec.validate_likelihood_compatibility(
                LinkFunction.IDENTITY, LogNormalPrior(std=1.0)
            )

    def test_validate_likelihood_compat_log_rejects_lognormal_prior(self):
        # Under the log link mu is a log-scale predictor, not a
        # response-scale mean, so LogNormalPrior stays rejected.
        with pytest.raises(ValueError, match="not compatible with link='log'"):
            LinkSpec.validate_likelihood_compatibility(
                LinkFunction.LOG, LogNormalPrior(std=1.0)
            )

    def test_validate_likelihood_compat_log_lognormal(self):
        LinkSpec.validate_likelihood_compatibility(
            LinkFunction.LOG, Prior("LogNormal", sigma=1)
        )

    def test_validate_likelihood_compat_log_unnamed_uses_class_name(self):
        class Unnamed:
            pass

        with pytest.raises(ValueError, match="'Unnamed' is not compatible"):
            LinkSpec.validate_likelihood_compatibility(LinkFunction.LOG, Unnamed())

    def test_validate_likelihood_compat_log_normal_raises(self):
        with pytest.raises(ValueError, match="not compatible"):
            LinkSpec.validate_likelihood_compatibility(
                LinkFunction.LOG, Prior("Normal", sigma=1)
            )


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
        sat2 = serialization.deserialize(d)
        assert isinstance(sat2, LogSaturation)
        assert sat2.to_dict() == d
        # The unscaled-input contract must survive a round trip.
        assert sat2.requires_unscaled_input is True

    def test_requires_unscaled_input_flag(self):
        """LogSaturation opts into raw inputs; the default does not."""
        assert LogSaturation().requires_unscaled_input is True
        assert LogisticSaturation().requires_unscaled_input is False

    def test_log_saturation_skips_channel_scaling(self, mock_pymc_sample):
        """A log-log MMM forces channel_scale to one (raw inputs)."""
        mmm = _make_mmm(link="log")
        X, y = _make_positive_panel()
        mmm.build_model(X, y)

        np.testing.assert_allclose(mmm.scalers["_channel"].values, 1.0)
        np.testing.assert_allclose(mmm.model["channel_scale"].get_value(), 1.0)

    def test_log_saturation_feeds_raw_channel_data(self, mock_pymc_sample):
        """With channel_scale == 1 the forward pass receives raw spend."""
        mmm = _make_mmm(link="log")
        X, y = _make_positive_panel()
        mmm.build_model(X, y)

        stored = np.sort(mmm.model["channel_data"].get_value().ravel())
        raw = np.sort(X[["C1", "C2"]].to_numpy().ravel())
        np.testing.assert_allclose(stored, raw)

    def test_logistic_saturation_keeps_channel_scaling(self, mock_pymc_sample):
        """A non-flagged saturation keeps data-derived channel scaling (!= 1)."""
        mmm = _make_mmm(link="identity")
        X, y = _make_positive_panel()
        mmm.build_model(X, y)
        assert not np.allclose(mmm.scalers["_channel"].values, 1.0)

    def test_log_saturation_warns_when_channel_scaling_overridden(
        self, mock_pymc_sample
    ):
        """Explicit channel scaling triggers a UserWarning (it is ignored)."""
        scaling = Scaling(
            target=DataDerivedScaling(method="max", dims=("country",)),
            channel=FixedScaling(dims=("country",), value=10.0),
        )
        mmm = _make_mmm(link="log", scaling=scaling)
        X, y = _make_positive_panel()
        with pytest.warns(UserWarning, match="channel scaling"):
            mmm.build_model(X, y)
        np.testing.assert_allclose(mmm.scalers["_channel"].values, 1.0)

    def test_log_saturation_no_warning_with_default_scaling(self, mock_pymc_sample):
        """Default (implicit) channel scaling is overridden silently."""
        mmm = _make_mmm(link="log")
        X, y = _make_positive_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            mmm.build_model(X, y)

    def test_log_saturation_elasticity_function_limit(self):
        """beta is the high-spend elasticity: d log f / d log x -> beta."""
        beta = 0.7
        x = np.array([1e5, 1e6])
        elasticity = beta * x / (1.0 + x)  # d/dlog x of beta*log(1+x)
        np.testing.assert_allclose(elasticity, beta, rtol=1e-4)

    @pytest.mark.slow
    def test_log_saturation_elasticity_recovery(self):
        """A real (small) fit recovers a known elasticity within tolerance."""
        rng = np.random.default_rng(42)
        n = 120
        dates = pd.date_range("2021-01-04", periods=n, freq="W-MON")
        x = rng.uniform(100, 2000, n)
        beta_true, alpha_true = 0.6, 1.2
        mu = alpha_true + beta_true * np.log1p(x)
        y = np.exp(mu) * np.exp(rng.normal(0, 0.05, n))

        df = pd.DataFrame({"date": dates, "C1": x})
        y_series = pd.Series(y, name="y")

        mmm = MMM(
            date_column="date",
            channel_columns=["C1"],
            adstock=GeometricAdstock(l_max=1),
            saturation=LogSaturation(),
            link="log",
        )
        mmm.fit(
            df,
            y_series,
            draws=300,
            tune=300,
            chains=2,
            cores=2,
            target_accept=0.9,
            random_seed=0,
            progressbar=False,
        )
        beta_hat = float(mmm.idata.posterior["saturation_beta"].mean())
        assert abs(beta_hat - beta_true) < 0.1


class TestBuildModelDeterministics:
    """Test that build_model creates the correct deterministic variables."""

    @pytest.mark.parametrize("link", ["identity", "log"])
    def test_build_model_has_total_media(self, link, mock_pymc_sample):
        mmm = _make_mmm(link=link)
        X, y = _make_positive_panel()
        mmm.build_model(X, y)
        assert "total_media_contribution_original_scale" in mmm.model.named_vars

    def test_build_model_identity_lognormal_raises(self, mock_pymc_sample):
        mmm = _make_mmm(
            link="identity",
            model_config={"likelihood": Prior("LogNormal", sigma=Prior("HalfNormal"))},
        )
        X, y = _make_positive_panel()
        with pytest.raises(ValueError, match="not compatible with link='identity'"):
            mmm.build_model(X, y)

    def test_build_model_identity_lognormal_prior_builds(self, mock_pymc_sample):
        # Regression test for #2858: LogNormalPrior takes mu on the response
        # scale, so it must be admitted under link='identity' and build an
        # observed variable without warnings.
        mmm = _make_mmm(
            link="identity",
            model_config={
                "likelihood": LogNormalPrior(
                    std=Prior("HalfNormal", sigma=0.5, dims=("country",)),
                    dims=("date", "country"),
                ),
            },
        )
        X, y = _make_positive_panel()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            mmm.build_model(X, y)
        assert "y" in mmm.model.named_vars
        assert np.isfinite(mmm.model.point_logps()["y"])

    def test_build_model_identity_lognormal_prior_negative_mu_ninf(
        self, mock_pymc_sample
    ):
        # A linear predictor that dips below zero must be rejected with -inf,
        # not folded to |mu| (#2858).
        mmm = _make_mmm(
            link="identity",
            model_config={
                "likelihood": LogNormalPrior(
                    std=Prior("HalfNormal", sigma=0.5, dims=("country",)),
                    dims=("date", "country"),
                ),
            },
        )
        X, y = _make_positive_panel()
        mmm.build_model(X, y)

        bad_point = mmm.model.initial_point()
        bad_point["intercept_contribution"] = np.full_like(
            bad_point["intercept_contribution"], -50.0
        )
        assert np.isneginf(mmm.model.point_logps(point=bad_point)["y"])

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

    def test_build_model_log_add_original_scale_succeeds(self, mock_pymc_sample):
        mmm = _make_mmm(link="log")
        X, y = _make_positive_panel()
        mmm.build_model(X, y)
        mmm.add_original_scale_contribution_variable(["channel_contribution"])
        assert "channel_contribution_original_scale" in mmm.model.named_vars

    @pytest.mark.parametrize("link", ["identity", "log"])
    def test_add_original_scale_contribution_variable(self, link, mock_pymc_sample):
        mmm = _make_mmm(link=link)
        X, y = _make_positive_panel()
        mmm.build_model(X, y)
        mmm.add_original_scale_contribution_variable(["channel_contribution"])
        assert "channel_contribution_original_scale" in mmm.model.named_vars


class TestDecomposition:
    """Counterfactual decomposition consistency for both link types."""

    @pytest.fixture()
    def identity_contributions(self, mock_pymc_sample):
        """Fit an identity-link MMM and return (mmm, contributions_df)."""
        mmm = _make_mmm(link="identity")
        X, y = _make_positive_panel()
        # Seed the mock fit so downstream sign/value assertions are
        # deterministic regardless of global RNG state / test ordering.
        mmm.fit(X, y, random_seed=42)
        return mmm, mmm.compute_mean_contributions_over_time()

    @pytest.fixture()
    def log_contributions(self, mock_pymc_sample):
        """Fit a log-link MMM and return (mmm, contributions_df)."""
        mmm = _make_mmm(link="log")
        X, y = _make_positive_panel()
        mmm.fit(X, y, random_seed=42)
        return mmm, mmm.compute_mean_contributions_over_time()

    def test_identity_returns_dataframe_with_expected_columns(
        self, identity_contributions
    ):
        """Identity-link decomposition returns a DataFrame with channel and intercept columns."""
        _mmm, df = identity_contributions
        assert isinstance(df, pd.DataFrame)
        assert "C1" in df.columns
        assert "C2" in df.columns
        assert "intercept" in df.columns

    def test_log_returns_dataframe_with_expected_columns(self, log_contributions):
        """Log-link decomposition returns a DataFrame with channel and intercept columns."""
        _mmm, df = log_contributions
        assert isinstance(df, pd.DataFrame)
        assert "C1" in df.columns
        assert "C2" in df.columns
        assert "intercept" in df.columns

    def test_log_channel_contributions_non_negative(self, log_contributions):
        """Channel counterfactual contributions are non-negative under log link."""
        _mmm, df = log_contributions
        for ch in ["C1", "C2"]:
            assert (df[ch] >= -1e-6).all(), f"Channel {ch} has negative contributions"

    def test_identity_and_log_produce_same_columns(
        self, identity_contributions, log_contributions
    ):
        """Both link types produce identical column sets."""
        _id_mmm, identity_df = identity_contributions
        _log_mmm, log_df = log_contributions
        assert set(identity_df.columns) == set(log_df.columns)

    def test_identity_counterfactual_sums_to_y_hat(self, identity_contributions):
        """Identity-link contributions sum exactly to y_hat at every row."""
        mmm, df = identity_contributions

        component_cols = [c for c in df.columns if c not in ("date", "country")]
        df["row_sum"] = df[component_cols].sum(axis=1)

        posterior = mmm.idata.posterior
        target_scale = mmm.idata.constant_data["target_scale"].squeeze(drop=True)
        mu = posterior["intercept_contribution"] + posterior[
            "channel_contribution"
        ].sum("channel")
        y_hat = (mu * target_scale).mean(("chain", "draw"))
        y_hat_df = y_hat.to_dataframe(name="expected").reset_index()

        merged = df.merge(y_hat_df, on=["date", "country"])
        np.testing.assert_allclose(
            merged["row_sum"].values, merged["expected"].values, rtol=1e-5
        )

    def test_log_counterfactual_does_not_sum_to_y_hat(self, log_contributions):
        """Log-link counterfactuals do not sum exactly to y_hat (interaction overlap)."""
        mmm, df = log_contributions

        component_cols = [c for c in df.columns if c not in ("date", "country")]
        df["row_sum"] = df[component_cols].sum(axis=1)

        posterior = mmm.idata.posterior
        target_scale = mmm.idata.constant_data["target_scale"].squeeze(drop=True)
        y_hat = (np.exp(posterior["mu"]) * target_scale).mean(("chain", "draw"))
        y_hat_df = y_hat.to_dataframe(name="expected").reset_index()

        merged = df.merge(y_hat_df, on=["date", "country"])
        assert not np.allclose(
            merged["row_sum"].values, merged["expected"].values, rtol=1e-5
        ), "Log-link counterfactuals should NOT sum exactly to y_hat"

    def test_log_counterfactual_intercept_positive(self, log_contributions):
        """Removing the intercept should reduce y_hat, so its counterfactual is positive."""
        _mmm, df = log_contributions
        assert (df["intercept"] > 0).all(), (
            "Intercept counterfactual should be positive"
        )

    def test_identity_channel_contributions_non_negative(self, identity_contributions):
        """Channel contributions are non-negative under identity link (adstock + saturation >= 0)."""
        _mmm, df = identity_contributions
        for ch in ["C1", "C2"]:
            assert (df[ch] >= -1e-6).all(), (
                f"Channel {ch} has unexpected negative values"
            )

    def test_identity_dataset_returns_xr_dataset(self, mock_pymc_sample):
        """Identity-link dataset has chain and draw dims."""
        mmm = _make_mmm(link="identity")
        X, y = _make_positive_panel()
        mmm.fit(X, y)
        ds = mmm.compute_counterfactual_contributions_dataset()
        assert isinstance(ds, xr.Dataset)
        assert "chain" in ds.dims
        assert "draw" in ds.dims
        assert "C1" in ds.data_vars
        assert "intercept" in ds.data_vars

    def test_log_dataset_returns_xr_dataset(self, mock_pymc_sample):
        """Log-link dataset has chain and draw dims."""
        mmm = _make_mmm(link="log")
        X, y = _make_positive_panel()
        mmm.fit(X, y)
        ds = mmm.compute_counterfactual_contributions_dataset()
        assert isinstance(ds, xr.Dataset)
        assert "chain" in ds.dims
        assert "draw" in ds.dims
        assert "C1" in ds.data_vars
        assert "intercept" in ds.data_vars

    def test_dataset_mean_matches_dataframe(self, identity_contributions):
        """Averaging the dataset over (chain, draw) reproduces the DataFrame output."""
        mmm, df = identity_contributions
        ds = mmm.compute_counterfactual_contributions_dataset()
        df_from_ds = ds.mean(("chain", "draw")).to_dataframe().reset_index()

        component_cols = [c for c in df.columns if c not in ("date", "country")]
        for col in component_cols:
            np.testing.assert_allclose(
                df_from_ds[col].values,
                df[col].values,
                rtol=1e-6,
            )

    def test_log_dataset_has_expected_vars(self, log_contributions):
        """Dataset variable names match the DataFrame component columns."""
        mmm, df = log_contributions
        ds = mmm.compute_counterfactual_contributions_dataset()
        component_cols = {c for c in df.columns if c not in ("date", "country")}
        assert set(ds.data_vars) == component_cols


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


class TestBudgetOptimizerLogLog:
    """Budget optimization must keep working under the log link (channel_scale == 1)."""

    @pytest.mark.slow
    def test_optimize_budget_log_log_runs_and_conserves_budget(self):
        """A log-log model optimizes without double-scaling and conserves the budget.

        Uses a real (small) fit rather than ``mock_pymc_sample`` because a
        mocked posterior draws coefficients from the (wide) prior, which sends
        ``exp(mu)`` to extreme values and makes the optimization landscape
        ill-conditioned.  A genuine fit anchors the coefficients to the data
        and exercises the realistic ``channel_scale == 1`` path.
        """
        rng = np.random.default_rng(7)
        n = 80
        dates = pd.date_range("2021-01-04", periods=n, freq="W-MON")
        x1 = rng.uniform(100, 1000, n)
        x2 = rng.uniform(100, 1000, n)
        mu = 1.0 + 0.5 * np.log1p(x1) + 0.2 * np.log1p(x2)
        y = np.exp(mu) * np.exp(rng.normal(0, 0.05, n))

        X = pd.DataFrame({"date": dates, "C1": x1, "C2": x2})
        y_series = pd.Series(y, name="y")

        mmm = MMM(
            date_column="date",
            channel_columns=["C1", "C2"],
            adstock=GeometricAdstock(l_max=1),
            saturation=LogSaturation(),
            link="log",
        )
        mmm.fit(
            X,
            y_series,
            draws=200,
            tune=200,
            chains=2,
            cores=2,
            random_seed=0,
            progressbar=False,
        )

        optimizable = BudgetOptimizerWrapper(
            model=mmm,
            start_date=X["date"].max() + pd.Timedelta(weeks=1),
            end_date=X["date"].max() + pd.Timedelta(weeks=5),
        )

        total_budget = 1000.0
        optimal_budgets, result = optimizable.optimize_budget(budget=total_budget)

        assert result.success
        assert isinstance(optimal_budgets, xr.DataArray)
        assert (optimal_budgets.values >= -1e-8).all()
        np.testing.assert_allclose(
            float(optimal_budgets.sum()), total_budget, rtol=1e-4
        )


class TestDecompositionRelationship:
    """The conserving (wrapper) and counterfactual (MMM) decompositions differ by design."""

    def test_conserving_vs_counterfactual_log(self, mock_pymc_sample):
        mmm = _make_mmm(link="log")
        X, y = _make_positive_panel()
        mmm.fit(X, y, random_seed=42)

        wrapper_ds = mmm.data.get_contributions(original_scale=True)
        posterior = mmm.idata.posterior
        target_scale = mmm.idata.constant_data["target_scale"].squeeze(drop=True)
        y_hat = np.exp(posterior["mu"]) * target_scale

        # (i) The conserving decomposition sums exactly to y_hat.
        xr.testing.assert_allclose(
            wrapper_ds["channels"].sum("channel") + wrapper_ds["baseline"], y_hat
        )

        # (ii) Its channel total equals the total media counterfactual lift.
        media_total = posterior["channel_contribution"].sum("channel")
        total_lift = (
            np.exp(posterior["mu"]) - np.exp(posterior["mu"] - media_total)
        ) * target_scale
        xr.testing.assert_allclose(wrapper_ds["channels"].sum("channel"), total_lift)

        # (iii) The per-channel counterfactual split differs from the
        # proportional (conserving) split: per-component counterfactuals
        # overlap on interactions and do not match the proportional shares.
        cf_ds = mmm.compute_counterfactual_contributions_dataset()
        channel_names = [
            str(c) for c in posterior["channel_contribution"].coords["channel"].values
        ]
        cf_channels_sum = sum(cf_ds[ch] for ch in channel_names)
        assert not np.allclose(
            cf_channels_sum.values,
            wrapper_ds["channels"].sum("channel").values,
        )


class TestCentralTendency:
    """Mean vs median log-link contributions."""

    def test_mean_equals_median_times_sigma_correction_log(self, mock_pymc_sample):
        """Mean contributions equal median contributions times exp(sigma**2/2)."""
        mmm = _make_mmm(link="log")
        X, y = _make_positive_panel()
        mmm.fit(X, y, random_seed=42)

        median_ds = mmm.compute_counterfactual_contributions_dataset(
            central_tendency="median"
        )
        mean_ds = mmm.compute_counterfactual_contributions_dataset(
            central_tendency="mean"
        )

        correction = np.exp(mmm.idata.posterior["y_sigma"] ** 2 / 2)
        for var in median_ds.data_vars:
            xr.testing.assert_allclose(mean_ds[var], median_ds[var] * correction)

    def test_central_tendency_noop_for_identity(self, mock_pymc_sample):
        """Identity link: mean and median contributions are identical."""
        mmm = _make_mmm(link="identity")
        X, y = _make_positive_panel()
        mmm.fit(X, y, random_seed=42)

        median_ds = mmm.compute_counterfactual_contributions_dataset(
            central_tendency="median"
        )
        mean_ds = mmm.compute_counterfactual_contributions_dataset(
            central_tendency="mean"
        )
        for var in median_ds.data_vars:
            xr.testing.assert_allclose(mean_ds[var], median_ds[var])

    def test_mean_correction_raises_without_sampled_sigma(self, mock_pymc_sample):
        """A clear error is raised when the likelihood sigma is not in the posterior."""
        mmm = _make_mmm(link="log")
        X, y = _make_positive_panel()
        mmm.fit(X, y, random_seed=42)
        del mmm.idata.posterior["y_sigma"]
        with pytest.raises(ValueError, match="sampled likelihood scale"):
            mmm.compute_counterfactual_contributions_dataset(central_tendency="mean")

    def test_mean_correction_is_deprecated_on_the_log_link(self):
        posterior = xr.Dataset(
            {"y_sigma": xr.DataArray([[0.5]], dims=("chain", "date"))}
        )
        with pytest.warns(DeprecationWarning, match="use to_mean_scale"):
            factor = LogLinkSpec().mean_correction(posterior)
        xr.testing.assert_allclose(factor, np.exp(posterior["y_sigma"] ** 2 / 2))

    def test_mean_correction_refuses_under_identity(self):
        # It used to return 1.0 here, which is the bug #2834 is about. A
        # deprecated method must not keep handing back a known-wrong number.
        with pytest.raises(ValueError, match="depends on the likelihood"):
            IdentityLinkSpec().mean_correction(xr.Dataset())


class TestTruncatedNormalMeanCorrection:
    """Identity link: TruncatedNormal shifts E[y] off mu, so contributions move."""

    @pytest.fixture(scope="class")
    def mmm(self, mock_pymc_sample) -> MMM:
        """Fit once for the class: the tests only read from the model."""
        mmm = _make_mmm(
            link="identity",
            model_config={
                "likelihood": Prior(
                    "TruncatedNormal", lower=0.0, sigma=Prior("HalfNormal", sigma=1)
                )
            },
        )
        X, y = _make_positive_panel()
        mmm.fit(X, y, random_seed=42)
        return mmm

    def test_mu_is_registered_under_identity(self, mmm):
        assert "mu" in mmm.idata.posterior

    def test_mu_keeps_the_predictor_dim_order_under_identity(self, mmm):
        """ "date" stays last: the identity branch does not transpose.

        The log branch moves "date" to the front, so a reader who assumes the
        log ordering under identity indexes the wrong axis.
        """
        assert mmm.idata.posterior["mu"].dims == ("chain", "draw", "country", "date")

    def test_mean_scale_parts_sum_to_the_truncated_mean(self, mmm):
        """The corrected parts reconcile against E[y], which is the #2834 ask.

        The per-term assertions above pin where the offset lands; this pins the
        property a reader actually reads the decomposition for.
        """
        mean_ds = mmm.compute_counterfactual_contributions_dataset(
            central_tendency="mean"
        )
        total = sum(mean_ds[var] for var in mean_ds.data_vars)

        posterior = mmm.idata.posterior
        target_scale = mmm.idata.constant_data["target_scale"].squeeze(drop=True)
        mu = posterior["mu"]
        sigma = posterior["y_sigma"]
        expected = xr.apply_ufunc(
            lambda m, s: stats.truncnorm.mean((0.0 - m) / s, np.inf, loc=m, scale=s),
            mu,
            sigma,
        )

        xr.testing.assert_allclose(
            total, (expected * target_scale).transpose(*total.dims)
        )

    def test_offset_matches_closed_form_truncated_mean(self, mmm):
        median_ds = mmm.compute_counterfactual_contributions_dataset(
            central_tendency="median"
        )
        mean_ds = mmm.compute_counterfactual_contributions_dataset(
            central_tendency="mean"
        )

        posterior = mmm.idata.posterior
        target_scale = mmm.idata.constant_data["target_scale"].squeeze(drop=True)
        mu = posterior["mu"]
        sigma = posterior["y_sigma"]
        expected = xr.apply_ufunc(
            lambda m, s: (
                stats.truncnorm.mean((0.0 - m) / s, np.inf, loc=m, scale=s) - m
            ),
            mu,
            sigma,
        )

        xr.testing.assert_allclose(
            mean_ds["intercept"], median_ds["intercept"] + expected * target_scale
        )

    def test_components_other_than_the_baseline_are_untouched(self, mmm):
        median_ds = mmm.compute_counterfactual_contributions_dataset(
            central_tendency="median"
        )
        mean_ds = mmm.compute_counterfactual_contributions_dataset(
            central_tendency="mean"
        )
        for var in median_ds.data_vars:
            if var == "intercept":
                continue
            xr.testing.assert_allclose(mean_ds[var], median_ds[var])

    def test_offset_is_finite_and_positive_for_negative_mu(self):
        # The offset must stay well behaved where mu <= 0, which is the case
        # the ratio form could not express.
        posterior = xr.Dataset(
            {
                "mu": xr.DataArray([[-2.0, -0.5, 0.0, 3.0]], dims=("chain", "date")),
                "y_sigma": xr.DataArray([[1.0, 1.0, 1.0, 1.0]], dims=("chain", "date")),
            }
        )
        likelihood = Prior("TruncatedNormal", lower=0, sigma=1)
        offset = IdentityLinkSpec()._truncation_offset(posterior, likelihood, "y")

        expected = [
            stats.truncnorm.mean((0.0 - m) / 1.0, np.inf, loc=m, scale=1.0) - m
            for m in (-2.0, -0.5, 0.0, 3.0)
        ]
        np.testing.assert_allclose(offset.values[0], expected)
        assert np.all(np.isfinite(offset.values))
        assert np.all(offset.values > 0)

    def test_offset_matches_the_analytic_half_normal_value(self):
        # At mu = 0 with lower = 0 the truncated normal is a half normal, whose
        # mean is sigma * sqrt(2 / pi). Independent of scipy's truncnorm.
        posterior = xr.Dataset(
            {
                "mu": xr.DataArray([[0.0, 0.0]], dims=("chain", "date")),
                "y_sigma": xr.DataArray([[1.0, 2.5]], dims=("chain", "date")),
            }
        )
        likelihood = Prior("TruncatedNormal", lower=0, sigma=1)
        offset = IdentityLinkSpec()._truncation_offset(posterior, likelihood, "y")
        np.testing.assert_allclose(
            offset.values[0], np.array([1.0, 2.5]) * np.sqrt(2 / np.pi)
        )

    def test_offset_stays_finite_far_below_the_truncation_point(self):
        # The textbook phi/Phi ratio returns nan from about ten sigma out. The
        # identity link puts no bound on mu, so this has to hold.
        posterior = xr.Dataset(
            {
                "mu": xr.DataArray([[-10.0, -40.0, -100.0]], dims=("chain", "date")),
                "y_sigma": xr.DataArray([[1.0, 1.0, 1.0]], dims=("chain", "date")),
            }
        )
        likelihood = Prior("TruncatedNormal", lower=0, sigma=1)
        offset = IdentityLinkSpec()._truncation_offset(posterior, likelihood, "y")
        assert np.all(np.isfinite(offset.values))
        # E[y] sits just above the truncation point, so the offset is about -mu.
        np.testing.assert_allclose(
            offset.values[0], [10.098093, 40.024969, 100.009998], rtol=1e-5
        )

    def test_two_sided_truncation_matches_scipy(self):
        posterior = xr.Dataset(
            {
                "mu": xr.DataArray([[-1.0, 0.5, 4.0]], dims=("chain", "date")),
                "y_sigma": xr.DataArray([[1.0, 2.0, 1.5]], dims=("chain", "date")),
            }
        )
        likelihood = Prior("TruncatedNormal", lower=0, upper=5, sigma=1)
        offset = IdentityLinkSpec()._truncation_offset(posterior, likelihood, "y")

        mus = np.array([-1.0, 0.5, 4.0])
        sigmas = np.array([1.0, 2.0, 1.5])
        expected = [
            stats.truncnorm.mean((0.0 - m) / s, (5.0 - m) / s, loc=m, scale=s) - m
            for m, s in zip(mus, sigmas, strict=True)
        ]
        np.testing.assert_allclose(offset.values[0], expected)

    def test_upper_only_truncation_matches_scipy(self):
        posterior = xr.Dataset(
            {
                "mu": xr.DataArray([[0.0, 4.0, 60.0]], dims=("chain", "date")),
                "y_sigma": xr.DataArray([[1.0, 1.0, 1.0]], dims=("chain", "date")),
            }
        )
        likelihood = Prior("TruncatedNormal", upper=3, sigma=1)
        offset = IdentityLinkSpec()._truncation_offset(posterior, likelihood, "y")

        expected = [
            stats.truncnorm.mean(-np.inf, (3.0 - m) / 1.0, loc=m, scale=1.0) - m
            for m in (0.0, 4.0, 60.0)
        ]
        np.testing.assert_allclose(offset.values[0], expected)
        assert np.all(np.isfinite(offset.values))

    @pytest.mark.parametrize(
        "inner",
        [
            Prior("TruncatedNormal", lower=0, sigma=1),
            Prior("Normal", sigma=1),
            Prior("StudentT", nu=3, sigma=1),
        ],
    )
    def test_censored_wrapper_raises_rather_than_using_the_wrong_mean(self, inner):
        # Censoring piles mass at the bounds, so E[y] != mu even for the
        # response-scale names the wrapper resolves to.
        posterior = xr.Dataset(
            {
                "mu": xr.DataArray([[1.0]], dims=("chain", "date")),
                "y_sigma": xr.DataArray([[1.0]], dims=("chain", "date")),
            }
        )
        dataset = xr.Dataset({"intercept": posterior["mu"]})
        with pytest.raises(ValueError, match="wrapped likelihood"):
            IdentityLinkSpec().to_mean_scale(
                dataset, posterior, Censored(inner, lower=0), xr.DataArray(1.0)
            )

    def test_scaled_wrapper_is_named_once_in_the_rejection(self):
        """Scaled resolves to its own class name, so do not print it twice.

        Censored resolves to the name it holds, which reads well. Scaled does
        not, and naming both gave "Scaled holding 'Scaled'".
        """
        posterior = xr.Dataset({"mu": xr.DataArray([[1.0]], dims=("chain", "date"))})
        dataset = xr.Dataset({"intercept": posterior["mu"]})
        likelihood = Scaled(Prior("TruncatedNormal", lower=0, sigma=1), factor=2)

        with pytest.raises(ValueError, match=r"\(Scaled\)\. The wrapper"):
            IdentityLinkSpec().to_mean_scale(
                dataset, posterior, likelihood, xr.DataArray(1.0)
            )

    def test_fixed_sigma_is_used_instead_of_the_posterior(self):
        posterior = xr.Dataset({"mu": xr.DataArray([[0.0]], dims=("chain", "date"))})
        likelihood = Prior("TruncatedNormal", lower=0, sigma=2.0)
        offset = IdentityLinkSpec()._truncation_offset(posterior, likelihood, "y")
        np.testing.assert_allclose(offset.values[0], [2.0 * np.sqrt(2 / np.pi)])

    def test_two_sided_stays_finite_far_from_both_bounds(self):
        # The direct phi/Phi form returns -inf and nan here.
        posterior = xr.Dataset(
            {
                "mu": xr.DataArray([[39.0, 42.0, -42.0]], dims=("chain", "date")),
                "y_sigma": xr.DataArray([[1.0, 1.0, 1.0]], dims=("chain", "date")),
            }
        )
        likelihood = Prior("TruncatedNormal", lower=0, upper=1, sigma=1)
        offset = IdentityLinkSpec()._truncation_offset(posterior, likelihood, "y")
        assert np.all(np.isfinite(offset.values))
        expected = [
            stats.truncnorm.mean((0.0 - m) / 1.0, (1.0 - m) / 1.0, loc=m, scale=1.0) - m
            for m in (39.0, 42.0, -42.0)
        ]
        np.testing.assert_allclose(offset.values[0], expected)

    @pytest.mark.parametrize(
        ("lower", "upper"),
        [
            (0.0, np.array([5.0, 5.0])),
            (np.array([0.0, 0.0]), 5.0),
            (np.array([0.0, 0.0]), np.array([5.0, 5.0])),
            (np.array([-np.inf, 0.0]), 5.0),
        ],
    )
    def test_array_valued_bounds_are_handled(self, lower, upper):
        # A vector bound used to reach `if np.isposinf(array)` and die on the
        # ambiguous truth value instead of computing an offset.
        posterior = xr.Dataset(
            {
                "mu": xr.DataArray([[0.5, 2.0]], dims=("chain", "date")),
                "y_sigma": xr.DataArray([[1.0, 1.0]], dims=("chain", "date")),
            }
        )
        likelihood = Prior("TruncatedNormal", lower=lower, upper=upper, sigma=1)
        offset = IdentityLinkSpec()._truncation_offset(posterior, likelihood, "y")

        lows = np.broadcast_to(lower, (2,))
        highs = np.broadcast_to(upper, (2,))
        expected = [
            stats.truncnorm.mean(lo - m, hi - m, loc=m, scale=1.0) - m
            for lo, hi, m in zip(lows, highs, (0.5, 2.0), strict=True)
        ]
        np.testing.assert_allclose(offset.values[0], expected)


class TestMeanScaleFactor:
    """The factor-only entry point used where a scale is folded in."""

    @staticmethod
    def _posterior():
        return xr.Dataset(
            {"y_sigma": xr.DataArray([[0.5, 0.5]], dims=("chain", "date"))}
        )

    def test_identity_normal_is_one(self):
        factor = IdentityLinkSpec().mean_scale_factor(
            self._posterior(), Prior("Normal", sigma=1)
        )
        assert float(factor) == 1.0

    def test_identity_truncated_normal_refuses(self):
        with pytest.raises(ValueError, match="is an offset, not a factor"):
            IdentityLinkSpec().mean_scale_factor(
                self._posterior(), Prior("TruncatedNormal", lower=0, sigma=1)
            )

    def test_log_returns_the_lognormal_ratio(self):
        posterior = self._posterior()
        factor = LogLinkSpec().mean_scale_factor(posterior, Prior("LogNormal", sigma=1))
        xr.testing.assert_allclose(factor, np.exp(posterior["y_sigma"] ** 2 / 2))

    # The factor entry point has to reject and warn about exactly what
    # to_mean_scale does. Returning 1.0 for any of these is the bug this
    # branch fixes, moved to the other caller.

    @pytest.mark.parametrize(
        "inner",
        [
            Prior("Normal", sigma=1),
            Prior("TruncatedNormal", lower=0, sigma=1),
            Prior("StudentT", nu=3, sigma=1),
        ],
    )
    def test_censored_wrapper_refuses(self, inner):
        with pytest.raises(ValueError, match="wrapped likelihood"):
            IdentityLinkSpec().mean_scale_factor(
                self._posterior(), Censored(inner, lower=0)
            )

    def test_studentt_at_or_below_one_refuses(self):
        with pytest.raises(ValueError, match="no mean when nu <= 1"):
            IdentityLinkSpec().mean_scale_factor(
                self._posterior(), Prior("StudentT", nu=1, sigma=1)
            )

    def test_unknown_likelihood_warns(self):
        with pytest.warns(UserWarning, match="No mean correction is known"):
            factor = IdentityLinkSpec().mean_scale_factor(
                self._posterior(), Prior("Weibull", alpha=1, beta=1)
            )
        assert float(factor) == 1.0

    def test_sampled_nu_message_counts_the_offending_draws(self):
        posterior = xr.Dataset(
            {"y_nu": xr.DataArray([[0.4, 0.9, 6.0]], dims=("chain", "date"))}
        )
        with pytest.raises(ValueError, match="2 of 3 draws are at or below 1"):
            IdentityLinkSpec().mean_scale_factor(
                posterior, Prior("StudentT", nu=Prior("Gamma", mu=2, sigma=1), sigma=1)
            )

    def test_missing_baseline_term_raises(self):
        posterior = xr.Dataset(
            {
                "mu": xr.DataArray([[1.0]], dims=("chain", "date")),
                "y_sigma": xr.DataArray([[1.0]], dims=("chain", "date")),
            }
        )
        dataset = xr.Dataset(
            {"channel_1": xr.DataArray([[1.0]], dims=("chain", "date"))}
        )
        with pytest.raises(ValueError, match="'intercept' term, which is missing"):
            IdentityLinkSpec().to_mean_scale(
                dataset,
                posterior,
                Prior("TruncatedNormal", lower=0, sigma=1),
                xr.DataArray(1.0),
            )

    def test_raises_without_mu_in_posterior(self):
        posterior = xr.Dataset(
            {"y_sigma": xr.DataArray([[1.0]], dims=("chain", "date"))}
        )
        likelihood = Prior("TruncatedNormal", lower=0, sigma=1)
        with pytest.raises(ValueError, match="need 'mu' in the posterior"):
            IdentityLinkSpec()._truncation_offset(posterior, likelihood, "y")


class TestIdentityMeanScaleDispatch:
    """to_mean_scale dispatches on the likelihood, not only on the link."""

    @staticmethod
    def _dataset():
        return xr.Dataset(
            {
                "channel_1": xr.DataArray([[1.0, 2.0]], dims=("chain", "date")),
                "intercept": xr.DataArray([[3.0, 4.0]], dims=("chain", "date")),
            }
        )

    @staticmethod
    def _posterior():
        return xr.Dataset(
            {
                "mu": xr.DataArray([[0.5, 1.5]], dims=("chain", "date")),
                "y_sigma": xr.DataArray([[1.0, 1.0]], dims=("chain", "date")),
                "y_nu": xr.DataArray([[5.0, 6.0]], dims=("chain", "date")),
            }
        )

    @pytest.mark.parametrize(
        "likelihood",
        [
            Prior("Normal", sigma=1),
            Prior("Gamma", sigma=1),
            Prior("Laplace", b=1),
            Prior("InverseGamma", sigma=1),
        ],
    )
    def test_response_scale_likelihoods_are_a_noop(self, likelihood):
        dataset = self._dataset()
        out = IdentityLinkSpec().to_mean_scale(
            dataset, self._posterior(), likelihood, xr.DataArray(2.0)
        )
        xr.testing.assert_identical(out, dataset)

    def test_studentt_above_one_is_a_noop(self):
        dataset = self._dataset()
        out = IdentityLinkSpec().to_mean_scale(
            dataset,
            self._posterior(),
            Prior("StudentT", nu=3, sigma=1),
            xr.DataArray(2.0),
        )
        xr.testing.assert_identical(out, dataset)

    def test_studentt_with_fixed_nu_at_or_below_one_raises(self):
        with pytest.raises(ValueError, match="no mean when nu <= 1"):
            IdentityLinkSpec().to_mean_scale(
                self._dataset(),
                self._posterior(),
                Prior("StudentT", nu=1, sigma=1),
                xr.DataArray(2.0),
            )

    def test_studentt_with_sampled_nu_below_one_raises(self):
        posterior = self._posterior()
        posterior["y_nu"] = xr.DataArray([[0.4, 6.0]], dims=("chain", "date"))
        with pytest.raises(ValueError, match="no mean when nu <= 1"):
            IdentityLinkSpec().to_mean_scale(
                self._dataset(),
                posterior,
                Prior("StudentT", nu=Prior("Gamma", mu=2, sigma=1), sigma=1),
                xr.DataArray(2.0),
            )

    def test_non_response_scale_likelihood_refuses(self):
        # validate_likelihood_compatibility stops a model reaching here, but
        # the correction path rejects it on its own rather than warning: 'mu'
        # is not even in the units of the target.
        with pytest.raises(ValueError, match="on the log scale"):
            IdentityLinkSpec().to_mean_scale(
                self._dataset(),
                self._posterior(),
                Prior("LogNormal", sigma=1),
                xr.DataArray(2.0),
            )

    def test_unknown_likelihood_warns_and_is_a_noop(self):
        dataset = self._dataset()
        with pytest.warns(UserWarning, match="No mean correction is known"):
            out = IdentityLinkSpec().to_mean_scale(
                dataset,
                self._posterior(),
                Prior("Weibull", alpha=1, beta=1),
                xr.DataArray(2.0),
            )
        xr.testing.assert_identical(out, dataset)


class TestMuNameCollision:
    """Registering mu turns a name clash into a build-time error."""

    def test_effect_registering_mu_collides_under_identity(self):
        # Before mu was registered under the identity link this built, because
        # the branch only set mu_var.name. Now PyMC rejects the duplicate.
        class MuNamingEffect(MuEffect):
            def create_data(self, mmm) -> None:
                pass

            def create_effect(self, mmm):
                return pmd.Deterministic(
                    "mu", pmd.Normal("collider", dims=("date",)) * 0.0
                )

            def set_data(self, mmm, model, X) -> None:
                pass

        mmm = _make_mmm(link="identity", dims=None)
        mmm.mu_effects.append(MuNamingEffect())
        X, y = _make_positive_panel(countries=("A",))
        X = X.drop(columns=["country"])

        with pytest.raises(ValueError, match="Variable name mu already exists"):
            mmm.build_model(X, y)


class TestMuEffectsDecomposition:
    """mu_effects must appear in the counterfactual decomposition."""

    def test_linear_trend_effect_appears_in_log_decomposition(self, mock_pymc_sample):
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
        X, y = _make_positive_panel(countries=("A",))
        X = X.drop(columns=["country"])
        with pytest.warns(UserWarning, match="mu_effects"):
            mmm.fit(X, y, random_seed=42)

        df = mmm.compute_mean_contributions_over_time()
        assert "trend_effect" in df.columns


class TestOriginalScaleGuardLogLink:
    """per-component *_original_scale under log link warns; output var does not."""

    def test_log_warns_for_component(self, mock_pymc_sample):
        mmm = _make_mmm(link="log")
        X, y = _make_positive_panel()
        mmm.build_model(X, y)
        with pytest.warns(UserWarning, match="multiplicative factors"):
            mmm.add_original_scale_contribution_variable(["channel_contribution"])

    def test_log_no_component_warning_for_output_var(self, mock_pymc_sample):
        mmm = _make_mmm(link="log")
        X, y = _make_positive_panel()
        mmm.build_model(X, y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mmm.add_original_scale_contribution_variable([mmm.output_var])
        assert not any("multiplicative factors" in str(w.message) for w in caught)

    def test_identity_no_warning(self, mock_pymc_sample):
        mmm = _make_mmm(link="identity")
        X, y = _make_positive_panel()
        mmm.build_model(X, y)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            mmm.add_original_scale_contribution_variable(["channel_contribution"])


class _DataShiftEffect(MuEffect):
    """Minimal mu effect adding a settable per-date term to the predictor.

    Deliberately trivial: the point of the tests below is *where the effect's
    contribution shows up*, not what it computes.
    """

    prefix: str = "shift"

    def to_dict(self) -> dict:
        """Serialize the effect."""
        return {"prefix": self.prefix}

    def create_data(self, mmm) -> None:
        """Register the settable per-date term."""
        pmd.Data(
            f"{self.prefix}_data",
            np.zeros(len(mmm.model.coords["date"])),
            dims="date",
        )

    def set_data(self, mmm, model, X) -> None:
        """No prediction-time refresh needed for these tests."""

    def create_effect(self, mmm):
        """Add the term to the linear predictor."""
        return pmd.Deterministic(
            f"{self.prefix}_effect_contribution",
            mmm.model[f"{self.prefix}_data"],
        )


class TestTotalResponseDeterministic:
    """``total_response_original_scale``: the objective that sees mu effects."""

    @staticmethod
    def _built(link: str, with_effect: bool) -> MMM:
        mmm = _make_mmm(link=link)
        if with_effect:
            mmm.mu_effects.append(_DataShiftEffect())
        X, y = _make_positive_panel()
        mmm.build_model(X, y)
        return mmm

    @pytest.mark.parametrize("link", ["identity", "log"])
    def test_absent_without_mu_effects(self, link, mock_pymc_sample):
        """Plain media models keep their posterior unchanged."""
        mmm = self._built(link, with_effect=False)
        assert "total_response_original_scale" not in mmm.model.named_vars

    @pytest.mark.parametrize("link", ["identity", "log"])
    def test_registered_with_mu_effects(self, link, mock_pymc_sample):
        mmm = self._built(link, with_effect=True)
        assert "total_response_original_scale" in mmm.model.named_vars

    def test_log_matches_summed_response(self, mock_pymc_sample):
        """Cross-check against the independently built ``y_original_scale``."""
        mmm = self._built("log", with_effect=True)
        with mmm.model:
            total, per_date = pm.draw(
                [
                    mmm.model["total_response_original_scale"],
                    mmm.model["y_original_scale"],
                ],
                random_seed=0,
            )
        np.testing.assert_allclose(float(total), float(per_date.sum()), rtol=1e-10)

    def test_identity_sees_the_effect_where_the_media_objective_does_not(
        self, mock_pymc_sample
    ):
        """The gap this closes: under identity the media objective is blind.

        ``total_media_contribution_original_scale`` is built from the channel
        tensor alone, so moving a mu effect leaves it bit-identical while the
        response genuinely changes -- which is how a budget optimized against
        it undervalues whatever drives the effect.
        """
        mmm = self._built("identity", with_effect=True)
        names = [
            "total_response_original_scale",
            "total_media_contribution_original_scale",
        ]
        n_dates = len(mmm.model.coords["date"])

        with mmm.model:
            before = pm.draw([mmm.model[n] for n in names], random_seed=0)
            pm.set_data(
                {"shift_data": np.full(n_dates, 0.5)},
                model=mmm.model,
            )
            after = pm.draw([mmm.model[n] for n in names], random_seed=0)

        assert float(before[0]) != float(after[0])
        assert float(before[1]) == float(after[1])


class TestLikelihoodSupport:
    """Target values outside the likelihood support are rejected at build time.

    See issue #2835.  The likelihood observes ``target / target_scale``, so
    every case here is stated in terms of that ratio rather than the target.
    """

    GAMMA = Prior("Gamma", sigma=Prior("HalfNormal", sigma=1), dims=("date",))

    def test_negative_scaled_target_raises(self, mock_pymc_sample):
        """One negative value, positive scale, so the ratio leaves the support."""
        mmm = _make_mmm(dims=None, model_config={"likelihood": self.GAMMA})
        X, y = _make_panel_with_target([-5.0, 3.0, 7.0, 2.0, 9.0, 4.0, 6.0, 8.0])
        with pytest.raises(ValueError, match="1 of 8 values"):
            mmm.build_model(X, y)

    def test_negative_target_with_negative_scale_builds(self, mock_pymc_sample):
        """An all-negative target scales to a positive ratio and must be allowed.

        ``DataDerivedScaling`` reduces with ``max``, not ``max(abs(...))``, so
        the scale is negative here and every observed value is positive.  A
        check written against the raw target would reject a model that fits.
        """
        mmm = _make_mmm(dims=None, model_config={"likelihood": self.GAMMA})
        X, y = _make_panel_with_target([-5.0, -3.0, -7.0, -2.0, -9.0, -4.0, -6.0, -8.0])
        mmm.build_model(X, y)

        assert float(mmm.scalers["_target"].values) < 0
        assert np.all(mmm.target_data_scaled.eval() > 0)
        logp = mmm.model.compile_logp()(mmm.model.initial_point())
        assert np.isfinite(logp)

    def test_all_zero_placeholder_target_is_skipped(self, mock_pymc_sample):
        """An all-zero target is the no-``y`` placeholder, not data to reject.

        ``fit`` and ``sample_prior_predictive`` substitute ``np.zeros`` when
        no target is given, and the scale is then zero, so every observed
        value is the clamped ``0.0``.  Rejecting that would make it impossible
        to build a model in order to look at its prior.
        """
        mmm = _make_mmm(dims=None, model_config={"likelihood": self.GAMMA})
        X, y = _make_panel_with_target([0.0] * 8)
        mmm.build_model(X, y)

    def test_prior_predictive_without_y_builds(self, mock_pymc_sample):
        """The same placeholder path, reached the way a user reaches it.

        Only the build is under test.  Drawing from a ``Gamma`` prior under
        ``link='identity'`` can raise ``scale < 0`` because ``mu`` is
        unconstrained, which is the likelihood's own business and depends on
        the seed.  That failure is tolerated; the support check firing on the
        placeholder is not.
        """
        mmm = _make_mmm(dims=None, model_config={"likelihood": self.GAMMA})
        X, _ = _make_panel_with_target([1.0] * 8)
        try:
            mmm.sample_prior_predictive(X, samples=5)
        except ValueError as exc:
            assert "requires the observed target" not in str(exc)
        assert "y" in mmm.model.named_vars

    def test_multidimensional_target_is_checked(self, mock_pymc_sample):
        """The mask is over the full ``(date, country)`` grid, not one series."""
        mmm = _make_mmm(model_config={"likelihood": self.GAMMA})
        X, y = _make_positive_panel()
        y.iloc[0] = -1.0
        with pytest.raises(ValueError, match="1 of 16 values"):
            mmm.build_model(X, y)

    def test_degenerate_scale_names_the_scale(self, mock_pymc_sample):
        """A zero scale clamps a real target to 0.0; say so, not "bad values".

        Country ``A``'s target has a maximum of zero, so its scale is zero and
        every ratio is NaN or infinite.  ``build_model`` rewrites those to
        ``0.0``, which is outside ``Gamma``'s support, and the user-visible
        cause is the scale rather than anything in the target.
        """
        dates = pd.date_range("2025-01-06", periods=4, freq="W-MON")
        rng = np.random.default_rng(0)
        rows = [
            {
                "date": d,
                "country": c,
                "C1": rng.uniform(10, 100),
                "C2": rng.uniform(10, 100),
            }
            for d in dates
            for c in ("A", "B")
        ]
        X = pd.DataFrame(rows)
        y = pd.Series([-2.0, 50.0, -2.0, 50.0, -2.0, 50.0, 0.0, 50.0], name="y")

        per_country = DataDerivedScaling(method="max", dims=())
        mmm = _make_mmm(
            scaling=Scaling(target=per_country, channel=per_country),
            model_config={
                "likelihood": Prior(
                    "Gamma",
                    sigma=Prior("HalfNormal", sigma=1),
                    dims=("date", "country"),
                )
            },
        )
        with pytest.raises(ValueError) as excinfo:
            mmm.build_model(X, y)

        message = str(excinfo.value)
        assert "'target_scale' has a zero entry" in message
        assert "the scale is the cause" in message

    def test_target_zeros_are_not_blamed_on_the_scale(self, mock_pymc_sample):
        """Genuine zeros in the target look identical to the clamp's output.

        A target containing zero weeks produces exact ``0.0`` observations
        under a perfectly healthy scale, so attributing every all-zero
        violation to a degenerate scale sends the reader hunting for a zero
        entry that is not there.
        """
        mmm = _make_mmm(dims=None, model_config={"likelihood": self.GAMMA})
        X, y = _make_panel_with_target([0.0, 5.0, 10.0, 0.0, 8.0, 3.0, 7.0, 2.0])
        with pytest.raises(ValueError) as excinfo:
            mmm.build_model(X, y)

        message = str(excinfo.value)
        assert "no zero entry" in message
        assert "zeros in the target rather than a scaling artefact" in message

    def test_zero_attribution_names_both_without_the_scale(self):
        """The staticmethod has no scale to consult, so it asserts neither."""
        likelihood = Prior("Gamma", dims=("date",))
        with pytest.raises(ValueError) as excinfo:
            LinkSpec.validate_likelihood_support(likelihood, np.array([0.0, 1.0]))

        message = str(excinfo.value)
        assert (
            "Check the target for zeros and 'target_scale' for a zero entry" in message
        )

    @pytest.mark.parametrize(
        "dist_name, kwargs",
        [
            ("Gamma", {}),
            ("Beta", {}),
            ("TruncatedNormal", {"sigma": 1, "lower": 0, "upper": 5}),
        ],
    )
    def test_nan_is_never_silently_accepted(self, dist_name, kwargs):
        """`np.nan <= 0` is False, so a naive mask lets NaN through."""
        likelihood = Prior(dist_name, dims=("date",), **kwargs)
        with pytest.raises(ValueError):
            LinkSpec.validate_likelihood_support(likelihood, np.array([0.5, np.nan]))

    def test_numpy_truncation_bounds_are_honoured(self):
        """A numpy scalar bound is a bound.

        ``Prior`` rejects ``np.int64`` and ``np.float32`` outright, so
        ``np.float64`` is the only numpy bound reachable through it.  It
        happens to subclass ``float``, but the check is written against
        ``numbers.Real`` so it does not depend on that.
        """
        likelihood = Prior(
            "TruncatedNormal",
            sigma=1,
            lower=np.float64(0),
            upper=np.float64(5),
            dims=("date",),
        )
        with pytest.raises(ValueError, match=r"lower 0.0 and upper 5.0"):
            LinkSpec.validate_likelihood_support(likelihood, np.array([2.5, 6.0]))

    def test_boolean_truncation_bound_is_not_treated_as_a_number(self):
        """`isinstance(False, int)` is True; `lower=False` is not a bound."""
        likelihood = Prior("TruncatedNormal", sigma=1, lower=False, dims=("date",))
        LinkSpec.validate_likelihood_support(likelihood, np.array([-100.0, 100.0]))

    def test_unevaluable_observed_is_skipped(self):
        """A check that can break build_model is worse than no check."""

        class Unevaluable:
            def eval(self):
                raise RuntimeError("cannot evaluate")

        likelihood = Prior("Gamma", dims=("date",))
        LinkSpec.validate_likelihood_support(likelihood, Unevaluable())

    def test_log_link_still_rejects_the_raw_target(self, mock_pymc_sample):
        """``LogLinkSpec.validate_target`` is unchanged and fires first.

        The scaled target here is strictly positive (the scale is negative),
        so the support check would accept it.  The link-level rule is a
        separate decision and still rejects it.
        """
        mmm = _make_mmm(link="log", dims=None)
        X, y = _make_panel_with_target([-5.0, -3.0, -7.0, -2.0, -9.0, -4.0, -6.0, -8.0])
        with pytest.raises(ValueError, match="strictly positive when using link='log'"):
            mmm.build_model(X, y)

    def test_unbounded_likelihood_is_not_checked(self, mock_pymc_sample):
        """``Normal`` has no bound, so the same target builds fine."""
        mmm = _make_mmm(dims=None)
        X, y = _make_panel_with_target([-5.0, 3.0, 7.0, 2.0, 9.0, 4.0, 6.0, 8.0])
        mmm.build_model(X, y)

    def test_censored_wrapper_is_not_checked(self, mock_pymc_sample):
        """Censoring at zero is what makes a zero valid; do not reject it.

        ``_distribution_name`` unwraps to ``LogNormal``, so a naive lookup
        would apply LogNormal's positivity rule to exactly the zero-inflated
        data the wrapper exists for.
        """
        likelihood = Censored(
            Prior("LogNormal", sigma=Prior("HalfNormal", sigma=1), dims=("date",)),
            lower=0,
        )
        observed = np.array([0.0, 0.5, 1.0])
        LinkSpec.validate_likelihood_support(likelihood, observed)

    @pytest.mark.parametrize(
        "dist_name, observed, expected",
        [
            ("LogNormal", [1.0, -1.0], "strictly positive"),
            ("InverseGamma", [1.0, 0.0], "strictly positive"),
            ("Beta", [0.5, 1.5], r"strictly inside \(0, 1\)"),
            ("Poisson", [1.0, 1.5], "a non-negative integer"),
            ("NegativeBinomial", [1.0, -2.0], "a non-negative integer"),
        ],
    )
    def test_support_rules(self, dist_name, observed, expected):
        likelihood = Prior(dist_name, dims=("date",))
        with pytest.raises(ValueError, match=expected):
            LinkSpec.validate_likelihood_support(likelihood, np.array(observed))

    def test_truncated_normal_uses_its_bounds(self):
        likelihood = Prior("TruncatedNormal", sigma=1, lower=0, upper=5, dims=("date",))
        LinkSpec.validate_likelihood_support(likelihood, np.array([0.0, 2.5, 5.0]))
        with pytest.raises(ValueError, match=r"lower 0 and upper 5"):
            LinkSpec.validate_likelihood_support(likelihood, np.array([2.5, 6.0]))

    def test_truncated_normal_without_numeric_bounds_is_skipped(self):
        """A ``Prior`` bound has no single interval to report, so skip it."""
        likelihood = Prior(
            "TruncatedNormal", sigma=1, lower=Prior("Normal"), dims=("date",)
        )
        LinkSpec.validate_likelihood_support(likelihood, np.array([-100.0, 100.0]))

    def test_error_names_the_distribution_and_the_count(self):
        likelihood = Prior("Gamma", dims=("date",))
        with pytest.raises(ValueError) as excinfo:
            LinkSpec.validate_likelihood_support(
                likelihood, np.array([1.0, -1.0, -2.0, 3.0])
            )
        message = str(excinfo.value)
        assert "'Gamma'" in message
        assert "strictly positive" in message
        assert "2 of 4 values" in message
