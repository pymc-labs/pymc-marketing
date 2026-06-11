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
"""Tests for the BLP synthetic data generator."""

import numpy as np
import pytest

from pymc_marketing.customer_choice.synthetic_data import generate_blp_panel


class TestBasicOutput:
    def test_returns_dataframe_by_default(self):
        df = generate_blp_panel(T=5, J=3, K=2, L=2, random_seed=0)
        assert hasattr(df, "columns")
        expected_rows = 5 * (3 + 1)  # T * (J + outside)
        assert len(df) == expected_rows

    def test_return_truth_gives_tuple(self):
        result = generate_blp_panel(
            T=5, J=3, K=2, L=2, random_seed=0, return_truth=True
        )
        assert isinstance(result, tuple)
        _, truth = result
        assert isinstance(truth, dict)

    def test_required_columns_present(self):
        df = generate_blp_panel(T=5, J=3, K=2, L=2, random_seed=0)
        for col in ["region", "market", "period", "product", "share", "n", "price"]:
            assert col in df.columns
        for k in range(2):
            assert f"x_{k}" in df.columns
        for ell in range(2):
            assert f"z_{ell}" in df.columns

    def test_shares_sum_to_one_per_market(self):
        df = generate_blp_panel(T=10, J=4, K=2, L=2, random_seed=0)
        share_sums = df.groupby("market")["share"].sum()
        np.testing.assert_allclose(share_sums, 1.0, atol=1e-10)

    def test_outside_good_has_zero_price_and_chars(self):
        df = generate_blp_panel(T=5, J=3, K=2, L=2, random_seed=0)
        outside = df[df["product"] == "outside"]
        assert (outside["price"] == 0.0).all()
        assert (outside["x_0"] == 0.0).all()
        assert (outside["x_1"] == 0.0).all()
        assert (outside["z_0"] == 0.0).all()
        assert (outside["z_1"] == 0.0).all()


class TestParameterVariants:
    def test_custom_true_beta(self):
        beta = np.array([0.5, -0.3])
        _, truth = generate_blp_panel(
            T=5, J=3, K=2, L=2, true_beta=beta, random_seed=0, return_truth=True
        )
        np.testing.assert_array_equal(truth["beta"], beta)

    def test_wrong_true_beta_shape_raises(self):
        with pytest.raises(ValueError, match="true_beta must have shape"):
            generate_blp_panel(T=5, J=3, K=2, L=2, true_beta=np.array([1.0, 2.0, 3.0]))

    def test_custom_sigma_beta(self):
        sigma = np.array([0.3, 0.0])
        _, truth = generate_blp_panel(
            T=5, J=3, K=2, L=2, sigma_beta=sigma, random_seed=0, return_truth=True
        )
        np.testing.assert_array_equal(truth["sigma_beta"], sigma)

    def test_wrong_sigma_beta_shape_raises(self):
        with pytest.raises(ValueError, match="sigma_beta must have shape"):
            generate_blp_panel(T=5, J=3, K=2, L=2, sigma_beta=np.array([0.1]))

    def test_negative_sigma_beta_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            generate_blp_panel(T=5, J=3, K=2, L=2, sigma_beta=np.array([-0.1, 0.0]))


class TestMultiRegion:
    def test_multi_region_output_shape(self):
        df = generate_blp_panel(T=5, J=3, K=2, L=2, R_geo=3, random_seed=0)
        expected_rows = 3 * 5 * (3 + 1)  # R * T * (J + outside)
        assert len(df) == expected_rows
        assert df["region"].nunique() == 3

    def test_region_heterogeneity_produces_different_alphas(self):
        _, truth = generate_blp_panel(
            T=5,
            J=3,
            K=2,
            L=2,
            R_geo=3,
            region_heterogeneity=0.5,
            random_seed=0,
            return_truth=True,
        )
        assert not np.allclose(truth["alpha_r"], truth["alpha_r"][0])

    def test_no_heterogeneity_gives_identical_regions(self):
        _, truth = generate_blp_panel(
            T=5,
            J=3,
            K=2,
            L=2,
            R_geo=3,
            region_heterogeneity=0.0,
            random_seed=0,
            return_truth=True,
        )
        np.testing.assert_array_equal(truth["alpha_r"], truth["alpha_r"][0])
        for r in range(3):
            np.testing.assert_array_equal(truth["beta_r"][r], truth["beta"])


class TestReproducibility:
    def test_same_seed_same_output(self):
        df1 = generate_blp_panel(T=5, J=3, K=2, L=2, random_seed=42)
        df2 = generate_blp_panel(T=5, J=3, K=2, L=2, random_seed=42)
        np.testing.assert_array_equal(df1["share"].values, df2["share"].values)
        np.testing.assert_array_equal(df1["price"].values, df2["price"].values)

    def test_different_seed_different_output(self):
        df1 = generate_blp_panel(T=5, J=3, K=2, L=2, random_seed=0)
        df2 = generate_blp_panel(T=5, J=3, K=2, L=2, random_seed=1)
        assert not np.allclose(df1["share"].values, df2["share"].values)

    def test_generator_input_accepted(self):
        rng = np.random.default_rng(42)
        df = generate_blp_panel(T=5, J=3, K=2, L=2, random_seed=rng)
        assert len(df) > 0


class TestTruthDict:
    def test_truth_keys(self):
        _, truth = generate_blp_panel(
            T=5, J=3, K=2, L=2, random_seed=0, return_truth=True
        )
        expected_keys = {
            "alpha",
            "alpha_r",
            "beta",
            "beta_r",
            "sigma_alpha",
            "sigma_beta",
            "xi_j",
            "xi_tilde",
            "xi",
            "pi_0",
            "pi_z",
            "price_xi_corr",
            "sigma_eta",
            "true_shares",
            "price_array",
            "characteristics_array",
            "instruments_array",
            "characteristic_cols",
            "instrument_cols",
        }
        assert set(truth.keys()) == expected_keys

    def test_truth_array_shapes(self):
        T, J, K, L = 8, 4, 3, 2
        _, truth = generate_blp_panel(
            T=T, J=J, K=K, L=L, random_seed=0, return_truth=True
        )
        assert truth["alpha_r"].shape == (1,)
        assert truth["beta"].shape == (K,)
        assert truth["beta_r"].shape == (1, K)
        assert truth["xi_j"].shape == (J,)
        assert truth["xi_tilde"].shape == (1, T, J)
        assert truth["xi"].shape == (1, T, J)
        assert truth["true_shares"].shape == (1, T, J + 1)
        assert truth["price_array"].shape == (1, T, J)
        assert truth["characteristics_array"].shape == (1, T, J, K)
        assert truth["instruments_array"].shape == (1, T, J, L)

    def test_truth_shapes_multi_region(self):
        T, J, K, L, R = 5, 3, 2, 2, 3
        _, truth = generate_blp_panel(
            T=T, J=J, K=K, L=L, R_geo=R, random_seed=0, return_truth=True
        )
        assert truth["alpha_r"].shape == (R,)
        assert truth["beta_r"].shape == (R, K)
        assert truth["true_shares"].shape == (R, T, J + 1)

    def test_characteristic_and_instrument_col_names(self):
        _, truth = generate_blp_panel(
            T=5, J=3, K=3, L=4, random_seed=0, return_truth=True
        )
        assert truth["characteristic_cols"] == ["x_0", "x_1", "x_2"]
        assert truth["instrument_cols"] == ["z_0", "z_1", "z_2", "z_3"]
