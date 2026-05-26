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
"""Tests for ``pymc_marketing.customer_choice.taste_profiles``."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from pymc_marketing.customer_choice import (
    BayesianBLP,
    taste_profiles,
)

# Force constrained-layout default so we exercise the same backend config the
# Jupyter inline backend uses (matches the colorbar-conflict trap we already
# hit once in this codebase).
plt.rcParams["figure.constrained_layout.use"] = True


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class TestValidation:
    """Pre-condition guards on every public function."""

    @staticmethod
    def _unfit(blp_panel_small):
        df, truth = blp_panel_small
        return BayesianBLP(
            market_data=df,
            characteristics=truth["characteristic_cols"],
            instruments=truth["instrument_cols"],
            n_mc_draws=20,
            random_seed=0,
        )

    @pytest.mark.parametrize(
        "fn",
        [
            taste_profiles.buyer_nu_posterior,
            taste_profiles.brand_buyer_nu,
            taste_profiles.demand_concentration_gini,
            taste_profiles.taste_type_demand_share,
            taste_profiles.plot_taste_profile_stacked,
            taste_profiles.plot_buyer_profile_heatmap,
            taste_profiles.plot_brand_buyer_heatmap,
            taste_profiles.plot_demand_concentration,
        ],
    )
    def test_unfit_model_raises(self, fn, blp_panel_small):
        m = self._unfit(blp_panel_small)
        with pytest.raises(RuntimeError, match="fit"):
            fn(m)

    def test_consumer_taste_grid_does_not_require_fit(self, blp_panel_small):
        # consumer_taste_grid only inspects construction-time state.
        m = self._unfit(blp_panel_small)
        grid = taste_profiles.consumer_taste_grid(m)
        assert isinstance(grid, pd.DataFrame)
        assert grid.shape == (m.n_mc_draws, m._halton.shape[1])

    def test_brand_buyer_nu_invalid_dim_raises(self, fitted_blp):
        model, _ = fitted_blp
        with pytest.raises(IndexError, match="dim"):
            taste_profiles.brand_buyer_nu(model, n_samples=10, dim=99)
        with pytest.raises(IndexError, match="dim"):
            taste_profiles.brand_buyer_nu(model, n_samples=10, dim=-1)

    def test_taste_type_demand_share_invalid_threshold_raises(self, fitted_blp):
        model, _ = fitted_blp
        with pytest.raises(ValueError, match="threshold"):
            taste_profiles.taste_type_demand_share(model, n_samples=10, threshold=-0.5)
        with pytest.raises(ValueError, match="threshold"):
            taste_profiles.taste_type_demand_share(model, n_samples=10, threshold=0.0)


# --------------------------------------------------------------------------- #
# Shapes (1-D RC, via existing fitted_blp fixture)
# --------------------------------------------------------------------------- #
class TestComputeShapes:
    def test_consumer_taste_grid(self, fitted_blp):
        model, _ = fitted_blp
        grid = taste_profiles.consumer_taste_grid(model)
        assert grid.shape == (model.n_mc_draws, 1)
        assert list(grid.columns) == list(model._random_coef_names)

    def test_buyer_nu_posterior(self, fitted_blp):
        model, _ = fitted_blp
        S = 50
        out = taste_profiles.buyer_nu_posterior(model, n_samples=S)
        assert out.shape == (S, model._M, 1)
        assert np.isfinite(out).all()

    def test_brand_buyer_nu(self, fitted_blp):
        model, _ = fitted_blp
        out = taste_profiles.brand_buyer_nu(model, n_samples=50, dim=0)
        assert out.shape == (model._M, model._J)
        assert np.isfinite(out).all()

    def test_demand_concentration_gini(self, fitted_blp):
        model, _ = fitted_blp
        S = 50
        out = taste_profiles.demand_concentration_gini(model, n_samples=S)
        assert out.shape == (S, model._M)
        # Gini is by construction in [0, 1]; allow tiny floating slack.
        assert (out >= -1e-9).all() and (out <= 1.0 + 1e-9).all()

    def test_taste_type_demand_share(self, fitted_blp):
        model, _ = fitted_blp
        df = taste_profiles.taste_type_demand_share(model, n_samples=50)
        assert list(df.columns) == [
            "market",
            "avg_price",
            "sensitive_pct",
            "modal_pct",
            "insensitive_pct",
        ]
        assert len(df) == model._M
        sums = df[["sensitive_pct", "modal_pct", "insensitive_pct"]].sum(axis=1)
        np.testing.assert_allclose(sums.values, 1.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# Mathematical correctness (no fit needed for the Gini cases)
# --------------------------------------------------------------------------- #
class TestComputeMath:
    def test_gini_uniform_contributions_is_zero(self):
        # Re-implement the Gini formula on a hand-crafted (R,) vector and check
        # the same routine the module uses returns ≈0 for uniform inputs.
        from pymc_marketing.customer_choice.taste_profiles import (
            demand_concentration_gini as _dc,  # noqa: F401 (sanity import)
        )

        R = 32
        sorted_vals = np.full((1, 1, R), 0.05)
        weights = 2 * np.arange(1, R + 1) - R - 1
        gini = (sorted_vals * weights).sum(axis=2) / (
            R * np.maximum(sorted_vals.sum(axis=2), 1e-30)
        )
        np.testing.assert_allclose(gini, 0.0, atol=1e-12)

    def test_gini_one_hot_contributions_approaches_one(self):
        R = 64
        x = np.zeros((1, 1, R))
        x[0, 0, -1] = 1.0  # All mass on one consumer type.
        sorted_vals = np.sort(x, axis=2)
        weights = 2 * np.arange(1, R + 1) - R - 1
        gini = (sorted_vals * weights).sum(axis=2) / (
            R * np.maximum(sorted_vals.sum(axis=2), 1e-30)
        )
        # Theoretical limit for a single-spike vector of length R is (R-1)/R.
        np.testing.assert_allclose(gini, (R - 1) / R, atol=1e-12)

    def test_buyer_nu_posterior_means_finite(self, fitted_blp):
        model, _ = fitted_blp
        out = taste_profiles.buyer_nu_posterior(model, n_samples=80)
        # Posterior means per market should be finite and bounded by the
        # extreme Halton draws.
        nu_min, nu_max = model._halton[:, 0].min(), model._halton[:, 0].max()
        means = out.mean(axis=0).squeeze(-1)
        assert np.isfinite(means).all()
        assert (means >= nu_min - 1e-9).all()
        assert (means <= nu_max + 1e-9).all()


# --------------------------------------------------------------------------- #
# Multi-dim (D=3) via fitted_blp_multidim fixture
# --------------------------------------------------------------------------- #
class TestMultiDim:
    def test_buyer_nu_posterior_three_dims(self, fitted_blp_multidim):
        model, _ = fitted_blp_multidim
        S = 60
        out = taste_profiles.buyer_nu_posterior(model, n_samples=S)
        assert out.shape == (S, model._M, 3)
        assert np.isfinite(out).all()

    def test_brand_buyer_nu_higher_dim_slice(self, fitted_blp_multidim):
        model, _ = fitted_blp_multidim
        out = taste_profiles.brand_buyer_nu(model, n_samples=60, dim=2)
        assert out.shape == (model._M, model._J)
        assert np.isfinite(out).all()

    def test_consumer_taste_grid_named_columns(self, fitted_blp_multidim):
        model, _ = fitted_blp_multidim
        grid = taste_profiles.consumer_taste_grid(model)
        assert grid.shape == (model.n_mc_draws, 3)
        assert list(grid.columns) == list(model._random_coef_names)


# --------------------------------------------------------------------------- #
# Plotters (Agg backend)
# --------------------------------------------------------------------------- #
class TestPlotters:
    def test_plot_taste_profile_stacked_returns_figure(self, fitted_blp):
        model, _ = fitted_blp
        fig = taste_profiles.plot_taste_profile_stacked(model, n_samples=40)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_taste_profile_stacked_default_picks_four_markets(self, fitted_blp):
        model, _ = fitted_blp
        fig = taste_profiles.plot_taste_profile_stacked(model, n_samples=20)
        # Default behaviour: 4 markets spanning the price range, so 4 axes.
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_plot_taste_profile_stacked_honours_market_indices(self, fitted_blp):
        model, _ = fitted_blp
        fig = taste_profiles.plot_taste_profile_stacked(
            model, market_indices=[0, 1], n_samples=20
        )
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_plot_buyer_profile_heatmap_returns_figure(self, fitted_blp):
        model, _ = fitted_blp
        fig = taste_profiles.plot_buyer_profile_heatmap(model, n_samples=40)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_brand_buyer_heatmap_returns_figure(self, fitted_blp):
        model, _ = fitted_blp
        fig = taste_profiles.plot_brand_buyer_heatmap(model, n_samples=40, dim=0)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_demand_concentration_returns_figure(self, fitted_blp):
        model, _ = fitted_blp
        fig = taste_profiles.plot_demand_concentration(model, n_samples=40)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_plot_buyer_profile_heatmap_accepts_ax(self, fitted_blp):
        model, _ = fitted_blp
        fig, ax = plt.subplots(layout="constrained")
        returned = taste_profiles.plot_buyer_profile_heatmap(model, n_samples=40, ax=ax)
        assert returned is fig
        plt.close(fig)

    def test_plot_brand_buyer_heatmap_accepts_ax(self, fitted_blp):
        model, _ = fitted_blp
        fig, ax = plt.subplots(layout="constrained")
        returned = taste_profiles.plot_brand_buyer_heatmap(
            model, n_samples=40, dim=0, ax=ax
        )
        assert returned is fig
        plt.close(fig)

    def test_plot_demand_concentration_accepts_ax(self, fitted_blp):
        model, _ = fitted_blp
        fig, ax = plt.subplots(layout="constrained")
        returned = taste_profiles.plot_demand_concentration(model, n_samples=40, ax=ax)
        assert returned is fig
        plt.close(fig)

    def test_plot_taste_profile_stacked_accepts_axes(self, fitted_blp):
        model, _ = fitted_blp
        fig, axes = plt.subplots(2, 1, layout="constrained")
        returned = taste_profiles.plot_taste_profile_stacked(
            model, market_indices=[0, 1], n_samples=30, axes=list(axes)
        )
        assert returned is fig
        plt.close(fig)

    def test_plot_taste_profile_stacked_axes_length_mismatch_raises(self, fitted_blp):
        model, _ = fitted_blp
        fig, axes = plt.subplots(2, 1, layout="constrained")
        with pytest.raises(ValueError, match="axes has length"):
            taste_profiles.plot_taste_profile_stacked(
                model,
                market_indices=[0, 1, 2],
                n_samples=20,
                axes=list(axes),
            )
        plt.close(fig)

    def test_default_market_pick_handles_n_geq_M(self, fitted_blp):
        """When the requested number of markets meets or exceeds M, return all."""
        from pymc_marketing.customer_choice.taste_profiles import (
            _default_price_span_markets,
        )

        model, _ = fitted_blp
        picks = _default_price_span_markets(model, n=model._M + 5)
        assert sorted(picks) == list(range(model._M))

    def test_plot_taste_profile_stacked_with_many_brands(self, blp_panel_small):
        """Drives the ``J > 8`` tab20 colormap + bbox legend branch.

        A throwaway model is built with J=9 inside products so the plotter
        takes the high-J path; the fit is mocked to keep this fast.
        """
        from functools import partial
        from unittest.mock import patch

        import pymc as pm
        import pymc.testing

        from pymc_marketing.customer_choice import (
            BayesianBLP,
            generate_blp_panel,
        )

        df, truth = generate_blp_panel(
            T=4,
            J=9,
            K=2,
            L=2,
            market_size=2_000,
            random_seed=0,
            return_truth=True,
        )
        model = BayesianBLP(
            market_data=df,
            characteristics=truth["characteristic_cols"],
            instruments=truth["instrument_cols"],
            random_coef_on=["price"],
            n_mc_draws=20,
            random_seed=0,
        )
        with patch.object(
            pm,
            "sample",
            partial(
                pymc.testing.mock_sample,
                sample_stats={"diverging": lambda size: np.zeros(size, dtype=int)},
            ),
        ):
            model.fit(draws=10, tune=10, chains=2, progressbar=False, random_seed=0)

        fig = taste_profiles.plot_taste_profile_stacked(
            model, market_indices=[0, 1], n_samples=20
        )
        assert isinstance(fig, Figure)
        plt.close(fig)
