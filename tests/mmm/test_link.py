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
import pytest
import xarray as xr
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
from pymc_marketing.mmm.mmm import MMM, BudgetOptimizerWrapper
from pymc_marketing.mmm.scaling import DataDerivedScaling, FixedScaling, Scaling


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
