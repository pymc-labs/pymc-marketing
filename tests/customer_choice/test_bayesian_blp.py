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
"""Tests for ``BayesianBLP`` (aggregate-share random-coefficients logit)."""

import warnings

import numpy as np
import pytest
import xarray as xr

from pymc_marketing.customer_choice import BayesianBLP, generate_blp_panel


@pytest.fixture(scope="session")
def fitted_blp(blp_panel_small):
    """A small BayesianBLP fitted on ``blp_panel_small``.

    Session-scoped — the ~30s fit runs once for every test in this file.
    """
    df, truth = blp_panel_small
    model = BayesianBLP(
        market_data=df,
        characteristics=truth["characteristic_cols"],
        instruments=truth["instrument_cols"],
        random_coef_on=["price"],
        n_mc_draws=80,
        random_seed=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(
            nuts_sampler="numpyro",
            draws=200,
            tune=200,
            chains=2,
            progressbar=False,
            random_seed=0,
        )
    return model, truth


class TestBuild:
    def test_single_region_build_creates_expected_vars(self, blp_panel_small):
        df, truth = blp_panel_small
        m = BayesianBLP(
            market_data=df,
            characteristics=truth["characteristic_cols"],
            instruments=truth["instrument_cols"],
            n_mc_draws=50,
            random_seed=0,
        )
        m.build_model()
        named = set(m.model.named_vars)
        assert {
            "alpha",
            "beta",
            "alpha_r",
            "beta_r",
            "xi_j",
            "xi_tilde",
            "xi",
            "eta",
            "rho_price_xi",
            "sigma_eta",
            "sigma_xi",
            "sigma_xi_j",
            "pi_0",
            "pi_z",
            "sigma_random",
            "s_inside",
            "s_outside",
            "log_share_ratio",
        } <= named

    def test_multi_region_adds_hierarchical_vars(self, blp_panel_multi_region):
        df, truth = blp_panel_multi_region
        m = BayesianBLP(
            market_data=df,
            characteristics=truth["characteristic_cols"],
            instruments=truth["instrument_cols"],
            region_col="region",
            n_mc_draws=50,
            random_seed=0,
        )
        m.build_model()
        named = set(m.model.named_vars)
        assert {"alpha_pop", "beta_pop", "tau_alpha", "tau_beta"} <= named
        assert m.model.named_vars["alpha_r"].eval().shape == (3,)

    def test_no_instruments_warns_and_skips_first_stage(self, blp_panel_small):
        df, truth = blp_panel_small
        with pytest.warns(UserWarning, match="not identified"):
            m = BayesianBLP(
                market_data=df,
                characteristics=truth["characteristic_cols"],
                instruments=None,
                n_mc_draws=50,
                random_seed=0,
            )
        m.build_model()
        named = set(m.model.named_vars)
        assert "eta" not in named
        assert "rho_price_xi" not in named
        assert "pi_0" not in named
        assert "xi_tilde" in named

    def test_n_random_zero_collapses_share_integral(self, blp_panel_small):
        """With no random coefficients the model degenerates to logit."""
        df, truth = blp_panel_small
        m = BayesianBLP(
            market_data=df,
            characteristics=truth["characteristic_cols"],
            instruments=truth["instrument_cols"],
            random_coef_on=[],
            n_mc_draws=10,
            random_seed=0,
        )
        m.build_model()
        assert "sigma_random" not in m.model.named_vars
        # Shares still defined and bounded
        s_in = m.model.named_vars["s_inside"].eval()
        s_out = m.model.named_vars["s_outside"].eval()
        np.testing.assert_allclose(s_in.sum(axis=-1) + s_out, 1.0, atol=1e-6)


class TestValidation:
    def test_unknown_random_coef_raises(self, blp_panel_small):
        df, truth = blp_panel_small
        with pytest.raises(ValueError, match="bogus"):
            BayesianBLP(
                market_data=df,
                characteristics=truth["characteristic_cols"],
                random_coef_on=["bogus"],
            )

    def test_unknown_likelihood_raises(self, blp_panel_small):
        df, truth = blp_panel_small
        with pytest.raises(ValueError, match="likelihood"):
            BayesianBLP(
                market_data=df,
                characteristics=truth["characteristic_cols"],
                likelihood="badname",
            )

    def test_product_fixed_effects_false_not_implemented(self, blp_panel_small):
        df, truth = blp_panel_small
        with pytest.raises(NotImplementedError, match="product_fixed_effects"):
            BayesianBLP(
                market_data=df,
                characteristics=truth["characteristic_cols"],
                product_fixed_effects=False,
            )

    def test_missing_characteristic_column_raises(self, blp_panel_small):
        df, truth = blp_panel_small
        with pytest.raises(ValueError, match="x_0"):
            BayesianBLP(
                market_data=df.drop(columns=["x_0"]),
                characteristics=truth["characteristic_cols"],
            )

    def test_missing_outside_good_raises(self, blp_panel_small):
        df, truth = blp_panel_small
        df_no_outside = df[df["product"] != "outside"].copy()
        with pytest.raises(ValueError, match="outside_good"):
            BayesianBLP(
                market_data=df_no_outside,
                characteristics=truth["characteristic_cols"],
            )

    def test_uneven_market_rows_raises(self, blp_panel_small):
        df, truth = blp_panel_small
        df_bad = df.iloc[1:].copy()
        with pytest.raises(ValueError, match="rows"):
            BayesianBLP(
                market_data=df_bad,
                characteristics=truth["characteristic_cols"],
            )

    def test_min_share_floor_warns(self):
        df, truth = generate_blp_panel(
            T=10, J=3, K=2, L=2, market_size=2_000, random_seed=0, return_truth=True
        )
        df.loc[df.index[0], "share"] = 0.0
        df.loc[df.index[1:4], "share"] = df.loc[df.index[1:4], "share"] + (
            -df.loc[df.index[0], "share"] / 3
        )
        with pytest.warns(UserWarning, match="Floored"):
            BayesianBLP(
                market_data=df,
                characteristics=truth["characteristic_cols"],
                random_seed=0,
            )


class TestPriorPredictive:
    def test_shares_sum_to_one_in_prior(self, blp_panel_small):
        df, truth = blp_panel_small
        m = BayesianBLP(
            market_data=df,
            characteristics=truth["characteristic_cols"],
            instruments=truth["instrument_cols"],
            n_mc_draws=50,
            random_seed=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prior = m.sample_prior_predictive(samples=20)
        s_in = prior.prior["s_inside"].values
        s_out = prior.prior["s_outside"].values
        assert s_in.shape == (1, 20, m._M, m._J)
        np.testing.assert_allclose(s_in.sum(axis=-1) + s_out, 1.0, atol=1e-6)


class TestFitSmoke:
    def test_fit_runs_without_divergences(self, fitted_blp):
        model, _ = fitted_blp
        n_div = int(model.idata.sample_stats["diverging"].values.sum())
        assert n_div == 0, f"got {n_div} divergences"


class TestCounterfactual:
    def test_no_op_canary_reproduces_fitted_shares(self, fitted_blp):
        """The strongest correctness check: passing no price change must
        yield shares identical (up to MC noise) to the model's own
        s_inside/s_outside posterior. If this fails, ξ is being silently
        re-derived under the price-data Deterministic dependency.
        """
        model, _ = fitted_blp
        cf = model.counterfactual_shares(price_change=None)
        assert isinstance(cf, xr.Dataset)
        cf_inside_mean = cf["s_inside"].mean(dim="sample").values
        cf_outside_mean = cf["s_outside"].mean(dim="sample").values
        post_inside_mean = (
            model.idata.posterior["s_inside"].mean(dim=("chain", "draw")).values
        )
        post_outside_mean = (
            model.idata.posterior["s_outside"].mean(dim=("chain", "draw")).values
        )
        np.testing.assert_allclose(cf_inside_mean, post_inside_mean, atol=5e-3)
        np.testing.assert_allclose(cf_outside_mean, post_outside_mean, atol=5e-3)

    def test_empty_dict_is_no_op(self, fitted_blp):
        model, _ = fitted_blp
        cf = model.counterfactual_shares(price_change={})
        assert cf["s_inside"].shape[1:] == (model._M, model._J)
        assert cf["s_outside"].shape[1:] == (model._M,)

    def test_price_increase_reduces_own_share(self, fitted_blp):
        """Raising one product's price should lower its average share and
        raise the outside-good share — the most basic structural check.
        """
        model, _ = fitted_blp
        target = model._inside_products[0]
        baseline = model.counterfactual_shares(price_change=None)
        shocked = model.counterfactual_shares(price_change={target: 0.20})
        baseline_share = baseline["s_inside"].mean(dim="sample").values[:, 0].mean()
        shocked_share = shocked["s_inside"].mean(dim="sample").values[:, 0].mean()
        assert shocked_share < baseline_share, (shocked_share, baseline_share)
        baseline_outside = baseline["s_outside"].mean(dim="sample").values.mean()
        shocked_outside = shocked["s_outside"].mean(dim="sample").values.mean()
        assert shocked_outside > baseline_outside

    def test_array_price_change_accepted(self, fitted_blp):
        model, _ = fitted_blp
        new_price = model._price * 1.05
        cf = model.counterfactual_shares(price_change=new_price, n_samples=20)
        assert cf["s_inside"].shape == (20, model._M, model._J)

    def test_unknown_product_in_price_change_raises(self, fitted_blp):
        model, _ = fitted_blp
        with pytest.raises(ValueError, match="Unknown product"):
            model.counterfactual_shares(price_change={"sku_xyz": 0.1})

    def test_wrong_array_shape_raises(self, fitted_blp):
        model, _ = fitted_blp
        with pytest.raises(ValueError, match="shape"):
            model.counterfactual_shares(price_change=np.zeros((3, 3)))

    def test_unfit_model_raises(self, blp_panel_small):
        df, truth = blp_panel_small
        m = BayesianBLP(
            market_data=df,
            characteristics=truth["characteristic_cols"],
            instruments=truth["instrument_cols"],
            n_mc_draws=20,
            random_seed=0,
        )
        with pytest.raises(RuntimeError, match="fit"):
            m.counterfactual_shares()


class TestElasticities:
    def test_mean_shape_and_dims(self, fitted_blp):
        model, _ = fitted_blp
        e = model.elasticities()
        assert isinstance(e, xr.DataArray)
        assert e.dims == ("market", "share", "price")
        assert e.shape == (model._M, model._J, model._J)

    def test_samples_shape_and_dims(self, fitted_blp):
        model, _ = fitted_blp
        e = model.elasticities(at="samples", n_samples=30)
        assert e.dims == ("sample", "market", "share", "price")
        assert e.shape == (30, model._M, model._J, model._J)

    def test_own_price_negative_on_average(self, fitted_blp):
        """For each inside product, the diagonal elasticity averaged across
        markets should be negative (price up → own share down).
        """
        model, _ = fitted_blp
        e = model.elasticities()
        for j in range(model._J):
            own = e.values[:, j, j].mean()
            assert own < 0, f"product {j} own elasticity {own} is not negative"

    def test_cross_price_non_negative_on_average(self, fitted_blp):
        """Cross-price elasticities should be non-negative on average for
        substitutes (which is the case under the basic mixed logit).
        """
        model, _ = fitted_blp
        e = model.elasticities()
        for j in range(model._J):
            for k in range(model._J):
                if j == k:
                    continue
                cross = e.values[:, j, k].mean()
                assert cross >= -1e-3, (
                    f"cross elasticity ({j}, {k}) = {cross} is meaningfully negative"
                )

    def test_invalid_at_raises(self, fitted_blp):
        model, _ = fitted_blp
        with pytest.raises(ValueError, match="at"):
            model.elasticities(at="bogus")

    def test_unfit_model_raises(self, blp_panel_small):
        df, truth = blp_panel_small
        m = BayesianBLP(
            market_data=df,
            characteristics=truth["characteristic_cols"],
            instruments=truth["instrument_cols"],
            n_mc_draws=20,
            random_seed=0,
        )
        with pytest.raises(RuntimeError, match="fit"):
            m.elasticities()


class TestTimeCol:
    """Phase 1 time-aware behaviour: coord-aware preprocessing, counterfactual
    masking, and the (region, period) sort invariant. No model-graph changes:
    when ``time_col=None`` the build is bit-identical to the pre-time-aware
    behaviour and that is asserted in the back-compat test below.
    """

    def test_synthetic_panel_has_period_column(self):
        df, _ = generate_blp_panel(T=8, J=3, K=2, L=2, random_seed=0, return_truth=True)
        assert "period" in df.columns
        # period is integer 0..T-1, present on every row including outside
        assert df["period"].nunique() == 8
        assert df.groupby("market")["period"].nunique().eq(1).all()

    def test_time_col_builds_period_coord_and_indices(self):
        df, truth = generate_blp_panel(
            T=8, J=3, K=2, L=2, R_geo=2, random_seed=7, return_truth=True
        )
        m = BayesianBLP(
            market_data=df,
            characteristics=truth["characteristic_cols"],
            instruments=truth["instrument_cols"],
            region_col="region",
            time_col="period",
            n_mc_draws=20,
            random_seed=0,
        )
        assert m._T == 8
        assert m._periods == sorted(set(df["period"]))
        assert m._period_idx is not None
        assert m._period_idx.shape == (m._M,)
        assert m._market_to_period is m._period_idx
        assert m._market_to_region is m._region_idx
        # Lex sort by (region, period) — markets[r*T + t] is (regions[r], periods[t])
        for r in range(len(m._regions)):
            for t in range(m._T):
                cell = r * m._T + t
                assert m._region_idx[cell] == r
                assert m._period_idx[cell] == t
        m.build_model()
        assert "period" in m.coords
        assert m.coords["period"] == m._periods

    def test_time_col_none_keeps_period_coord_absent(self, blp_panel_small):
        """Backward-compat: with time_col=None the period coord is not added."""
        df, truth = blp_panel_small
        m = BayesianBLP(
            market_data=df,
            characteristics=truth["characteristic_cols"],
            instruments=truth["instrument_cols"],
            n_mc_draws=20,
            random_seed=0,
        )
        m.build_model()
        assert "period" not in m.coords
        assert m._periods is None
        assert m._period_idx is None

    def test_shuffle_invariance(self):
        """Permuting the input rows must not change the internal arrays."""
        df, truth = generate_blp_panel(
            T=8, J=3, K=2, L=2, R_geo=3, random_seed=11, return_truth=True
        )
        kwargs = dict(
            characteristics=truth["characteristic_cols"],
            instruments=truth["instrument_cols"],
            region_col="region",
            time_col="period",
            n_mc_draws=20,
            random_seed=0,
        )
        m_ref = BayesianBLP(market_data=df, **kwargs)
        df_shuf = df.sample(frac=1.0, random_state=99).reset_index(drop=True)
        m_shuf = BayesianBLP(market_data=df_shuf, **kwargs)
        np.testing.assert_array_equal(m_shuf._period_idx, m_ref._period_idx)
        np.testing.assert_array_equal(m_shuf._region_idx, m_ref._region_idx)
        np.testing.assert_array_equal(m_shuf._inside_share, m_ref._inside_share)
        np.testing.assert_array_equal(m_shuf._outside_share, m_ref._outside_share)
        np.testing.assert_array_equal(m_shuf._price, m_ref._price)
        np.testing.assert_array_equal(m_shuf._x, m_ref._x)
        assert m_shuf._markets == m_ref._markets

    def test_non_rectangular_panel_raises(self):
        df, truth = generate_blp_panel(
            T=6, J=3, K=2, L=2, R_geo=2, random_seed=0, return_truth=True
        )
        # Drop all rows for one (region, period) cell — panel is now ragged
        drop_market = df.loc[
            (df["region"] == "r0") & (df["period"] == 3), "market"
        ].iloc[0]
        df_ragged = df[df["market"] != drop_market].copy()
        with pytest.raises(ValueError, match="rectangular"):
            BayesianBLP(
                market_data=df_ragged,
                characteristics=truth["characteristic_cols"],
                instruments=truth["instrument_cols"],
                region_col="region",
                time_col="period",
                n_mc_draws=20,
                random_seed=0,
            )

    def test_periods_without_time_col_raises(self, blp_panel_small):
        df, truth = blp_panel_small
        m = BayesianBLP(
            market_data=df,
            characteristics=truth["characteristic_cols"],
            instruments=truth["instrument_cols"],
            n_mc_draws=20,
            random_seed=0,
        )
        with pytest.raises(ValueError, match="time_col was not set"):
            m._resolve_cell_mask(periods=[0, 1], regions=None)

    def test_unknown_period_raises(self):
        df, truth = generate_blp_panel(
            T=6, J=3, K=2, L=2, random_seed=0, return_truth=True
        )
        m = BayesianBLP(
            market_data=df,
            characteristics=truth["characteristic_cols"],
            instruments=truth["instrument_cols"],
            time_col="period",
            n_mc_draws=20,
            random_seed=0,
        )
        with pytest.raises(ValueError, match="Unknown period"):
            m._resolve_cell_mask(periods=[999], regions=None)

    def test_unknown_region_raises(self):
        df, truth = generate_blp_panel(
            T=6, J=3, K=2, L=2, R_geo=2, random_seed=0, return_truth=True
        )
        m = BayesianBLP(
            market_data=df,
            characteristics=truth["characteristic_cols"],
            instruments=truth["instrument_cols"],
            region_col="region",
            time_col="period",
            n_mc_draws=20,
            random_seed=0,
        )
        with pytest.raises(ValueError, match="Unknown region"):
            m._resolve_cell_mask(periods=None, regions=["does-not-exist"])

    def test_string_periods_arg_rejected(self):
        df, truth = generate_blp_panel(
            T=6, J=3, K=2, L=2, random_seed=0, return_truth=True
        )
        m = BayesianBLP(
            market_data=df,
            characteristics=truth["characteristic_cols"],
            instruments=truth["instrument_cols"],
            time_col="period",
            n_mc_draws=20,
            random_seed=0,
        )
        with pytest.raises(ValueError, match="not a single string"):
            m._resolve_cell_mask(periods="foo", regions=None)

    def test_ndarray_price_change_rejected_when_time_col_set(self):
        df, truth = generate_blp_panel(
            T=6, J=3, K=2, L=2, random_seed=0, return_truth=True
        )
        m = BayesianBLP(
            market_data=df,
            characteristics=truth["characteristic_cols"],
            instruments=truth["instrument_cols"],
            time_col="period",
            n_mc_draws=20,
            random_seed=0,
        )
        with pytest.raises(ValueError, match="time_col is set"):
            m._resolve_price_change(np.zeros((m._M, m._J)))

    def test_resolve_price_change_with_periods_dict(self):
        df, truth = generate_blp_panel(
            T=6, J=3, K=2, L=2, random_seed=0, return_truth=True
        )
        m = BayesianBLP(
            market_data=df,
            characteristics=truth["characteristic_cols"],
            instruments=truth["instrument_cols"],
            time_col="period",
            n_mc_draws=20,
            random_seed=0,
        )
        baseline = m._price.copy()
        new = m._resolve_price_change({"prod_0": 0.10}, periods=[2, 3], regions=None)
        # Outside the targeted periods, price[..., 0] is untouched
        outside_mask = ~np.isin(m._period_idx, [2, 3])
        np.testing.assert_array_equal(new[outside_mask, 0], baseline[outside_mask, 0])
        # Inside, price scales by (1 + 0.10)
        inside_mask = np.isin(m._period_idx, [2, 3])
        np.testing.assert_allclose(new[inside_mask, 0], baseline[inside_mask, 0] * 1.10)
        # Other products are untouched everywhere
        np.testing.assert_array_equal(new[:, 1:], baseline[:, 1:])

    def test_resolve_price_change_slice(self):
        df, truth = generate_blp_panel(
            T=8, J=3, K=2, L=2, random_seed=0, return_truth=True
        )
        m = BayesianBLP(
            market_data=df,
            characteristics=truth["characteristic_cols"],
            instruments=truth["instrument_cols"],
            time_col="period",
            n_mc_draws=20,
            random_seed=0,
        )
        # slice(start, stop) is inclusive of stop on coord labels
        new = m._resolve_price_change({"prod_0": 0.10}, periods=slice(2, 4))
        affected = np.isin(m._period_idx, [2, 3, 4])
        np.testing.assert_allclose(new[affected, 0], m._price[affected, 0] * 1.10)
        np.testing.assert_array_equal(new[~affected, 0], m._price[~affected, 0])

    def test_resolve_price_change_regions_only(self):
        df, truth = generate_blp_panel(
            T=4, J=3, K=2, L=2, R_geo=2, random_seed=0, return_truth=True
        )
        m = BayesianBLP(
            market_data=df,
            characteristics=truth["characteristic_cols"],
            instruments=truth["instrument_cols"],
            region_col="region",
            time_col="period",
            n_mc_draws=20,
            random_seed=0,
        )
        new = m._resolve_price_change({"prod_0": 0.10}, regions=["r0"])
        r0_cells = m._region_idx == 0
        r1_cells = m._region_idx == 1
        np.testing.assert_allclose(new[r0_cells, 0], m._price[r0_cells, 0] * 1.10)
        np.testing.assert_array_equal(new[r1_cells, 0], m._price[r1_cells, 0])

    def test_empty_periods_yields_no_op(self):
        df, truth = generate_blp_panel(
            T=4, J=3, K=2, L=2, random_seed=0, return_truth=True
        )
        m = BayesianBLP(
            market_data=df,
            characteristics=truth["characteristic_cols"],
            instruments=truth["instrument_cols"],
            time_col="period",
            n_mc_draws=20,
            random_seed=0,
        )
        new = m._resolve_price_change({"prod_0": 0.10}, periods=[])
        np.testing.assert_array_equal(new, m._price)
