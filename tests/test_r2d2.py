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

"""Tests for R2-D2-M2 prior decomposition."""

import copy

import numpy as np
import pymc as pm
import pymc.dims as pmd
import pytest
import xarray as xr
from pymc_extras.prior import Prior

from pymc_marketing.r2d2 import R2D2, R2D2Sigma, R2D2Split
from pymc_marketing.serialization import serialization


class TestR2D2Split:
    """Tests for R2D2Split lazy reference."""

    def test_dims_available_immediately(self):
        """Dims should be available immediately from decomposition.

        The key is the component name (lookup key), the value is the
        actual dimension name in the model coords.
        """
        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.4),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={"my_control": "control_dim"},  # key ≠ value
        )
        split = r2d2.split("my_control")  # lookup by component name
        # Returns the actual dimension name from the value
        assert split.dims == ("control_dim",)

    def test_create_variable_auto_builds(self):
        """create_variable should auto-build decomposition if needed."""
        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.4),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={"control": "control"},
        )

        ds = xr.Dataset(
            {"controls": (("obs", "control"), np.random.randn(10, 2))},
            coords={"obs": range(10), "control": ["a", "b"]},
        )

        with pm.Model(coords={"obs": range(10), "control": ["a", "b"]}) as _model:
            pmd.Data("controls", ds["controls"])

            # Should auto-build decomposition
            split = r2d2.split("control")
            var = split.create_variable("test_var")

            assert r2d2.built
            assert var is not None
            assert "test_var" in var.name

    def test_deepcopy_preserves_reference(self):
        """Deepcopy should preserve decomposition reference."""
        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.4),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={"control": "control"},
        )

        split1 = r2d2.split("control")
        split2 = copy.deepcopy(split1)

        # Both should reference the same decomposition
        assert split1.decomposition is split2.decomposition

    def test_split_raises_on_unknown_component(self):
        """split() should raise informative error for unknown component."""
        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.4),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={"control": "control", "fourier": "fourier_mode"},
        )

        # Should raise ValueError with informative message
        with pytest.raises(ValueError, match="Component 'unknown' not found"):
            r2d2.split("unknown")

        # Error should list available components
        with pytest.raises(ValueError, match=r"Available components.*control.*fourier"):
            r2d2.split("unknown")


class TestR2D2Sigma:
    """Tests for R2D2Sigma lazy reference."""

    def test_dims_is_empty_tuple(self):
        """Sigma dims should be empty tuple (scalar)."""
        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.4),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={"control": "control"},
        )
        sigma = r2d2.error_sigma
        assert sigma.dims == ()

    def test_create_variable_auto_builds(self):
        """create_variable should auto-build decomposition if needed."""
        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.4),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={"control": "control"},
        )

        ds = xr.Dataset(
            {"controls": (("obs", "control"), np.random.randn(10, 2))},
            coords={"obs": range(10), "control": ["a", "b"]},
        )

        with pm.Model(coords={"obs": range(10), "control": ["a", "b"]}):
            pmd.Data("controls", ds["controls"])

            # Should auto-build decomposition
            sigma = r2d2.error_sigma
            var = sigma.create_variable("sigma")

            assert r2d2.built
            assert var is not None

    def test_deepcopy_preserves_reference(self):
        """Deepcopy should preserve decomposition reference."""
        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.4),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={"control": "control"},
        )

        sigma1 = r2d2.error_sigma
        sigma2 = copy.deepcopy(sigma1)

        # Both should reference the same decomposition
        assert sigma1.decomposition is sigma2.decomposition

    def test_error_sigma_is_scalar(self):
        """Error sigma should be a scalar variable."""
        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.4),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={"control": "control"},
        )

        ds = xr.Dataset(
            {"controls": (("obs", "control"), np.random.randn(10, 2))},
            coords={"obs": range(10), "control": ["a", "b"]},
        )

        with pm.Model(coords={"obs": range(10), "control": ["a", "b"]}):
            pmd.Data("controls", ds["controls"])

            error_sigma = r2d2.error_sigma.create_variable("error_sigma")
            assert error_sigma.type.ndim == 0  # scalar


class TestR2D2:
    """Tests for R2D2."""

    def test_r2_must_be_prior(self):
        """r2 parameter must be a Prior instance."""
        with pytest.raises(TypeError, match="r2 must be a Prior"):
            R2D2(
                r2="Beta",  # string, not Prior
                total_sigma=Prior("LogNormal", mu=0, sigma=1),
                dims={"control": "control"},
            )

    def test_total_sigma_must_be_prior(self):
        """total_sigma parameter must be a Prior instance."""
        with pytest.raises(TypeError, match="total_sigma must be a Prior"):
            R2D2(
                r2=Prior("Beta", mu=0.8, sigma=0.4),
                total_sigma="LogNormal",  # string, not Prior
                dims={"control": "control"},
            )

    def test_total_sigma_must_be_scalar(self):
        """total_sigma must be scalar (no dims) per R2D2 paper."""
        with pytest.raises(ValueError, match="total_sigma must be a scalar"):
            R2D2(
                r2=Prior("Beta", mu=0.8, sigma=0.4),
                total_sigma=Prior("LogNormal", mu=0, sigma=1, dims="geo"),
                dims={"control": "control"},
            )

    def test_r2_must_be_scalar(self):
        """r2 must be scalar (no dims) per R2D2 paper."""
        with pytest.raises(ValueError, match="r2 must be a scalar"):
            R2D2(
                r2=Prior("Beta", mu=0.8, sigma=0.4, dims="geo"),
                total_sigma=Prior("LogNormal", mu=0, sigma=1),
                dims={"control": "control"},
            )

    def test_create_variable_once(self):
        """create_variable should only create variables once (cached)."""
        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.4),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={"control": "control"},
        )

        ds = xr.Dataset(
            {"controls": (("obs", "control"), np.random.randn(10, 2))},
            coords={"obs": range(10), "control": ["a", "b"]},
        )

        with pm.Model(coords={"obs": range(10), "control": ["a", "b"]}):
            pmd.Data("controls", ds["controls"])

            # Call create_variable multiple times
            error1 = r2d2.create_variable("r2d2")
            error2 = r2d2.create_variable("r2d2")

            # Should return the same cached object
            assert error1 is error2
            assert r2d2.built

    def test_create_variable_returns_tensor(self):
        """create_variable should return a tensor variable."""
        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.4),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={"control": "control"},
        )

        ds = xr.Dataset(
            {"controls": (("obs", "control"), np.random.randn(10, 2))},
            coords={"obs": range(10), "control": ["a", "b"]},
        )

        with pm.Model(coords={"obs": range(10), "control": ["a", "b"]}):
            pmd.Data("controls", ds["controls"])

            # Create variable
            error = r2d2.create_variable("r2d2")

            # Should be a tensor variable
            from pytensor.tensor.variable import TensorVariable

            assert isinstance(error, TensorVariable)

    def test_split_returns_pmd_normal(self):
        """Split create_variable should return pmd.Normal (xtensor-based)."""
        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.4),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={"control": "control"},
        )

        ds = xr.Dataset(
            {"controls": (("obs", "control"), np.random.randn(10, 2))},
            coords={"obs": range(10), "control": ["a", "b"]},
        )

        with pm.Model(coords={"obs": range(10), "control": ["a", "b"]}):
            pmd.Data("controls", ds["controls"])

            # Create split variable - returns pmd.Normal (xtensor-based)
            split = r2d2.split("control")
            var = split.create_variable("test_var")

            # Should be an xtensor variable (pmd.Normal returns xtensor)
            from pytensor.xtensor.type import XTensorVariable

            assert isinstance(var, XTensorVariable)

    def test_dims_rejects_non_string_values(self):
        """dims dict values must be strings."""
        with pytest.raises(TypeError, match=r"dim value for.*must be a string"):
            R2D2(
                r2=Prior("Beta", mu=0.8, sigma=0.4),
                total_sigma=Prior("LogNormal", mu=0, sigma=1),
                dims={"control": 123},  # int, not string
            )

    def test_splits_returns_copy(self):
        """splits property should return a copy of the internal dict."""
        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.4),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={"control": "control"},
        )

        ds = xr.Dataset(
            {"controls": (("obs", "control"), np.random.randn(10, 2))},
            coords={"obs": range(10), "control": ["a", "b"]},
        )

        with pm.Model(coords={"obs": range(10), "control": ["a", "b"]}):
            pmd.Data("controls", ds["controls"])
            r2d2.create_variable("r2d2")

            splits1 = r2d2.splits
            splits2 = r2d2.splits

            # Should be equal but not the same object
            assert splits1 == splits2
            assert splits1 is not splits2

    def test_split_dim_uses_variable_names(self):
        """split_dim coord should use actual variable names, not generic indices."""
        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.2),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={
                "control_vars": "control",
                "media": "channel",
            },
        )

        ds = xr.Dataset(
            {
                "controls": (("obs", "control"), np.random.randn(10, 3)),
                "channels": (("obs", "channel"), np.random.randn(10, 2)),
            },
            coords={
                "obs": range(10),
                "control": ["event_1", "event_2", "t"],
                "channel": ["x1", "x2"],
            },
        )

        with pm.Model(
            coords={
                "obs": range(10),
                "control": ["event_1", "event_2", "t"],
                "channel": ["x1", "x2"],
            }
        ) as model:
            pmd.Data("controls", ds["controls"])
            pmd.Data("channels", ds["channels"])
            r2d2.create_variable("r2d2")

            # split_dim should use actual variable names
            expected_names = ["event_1", "event_2", "t", "x1", "x2"]
            actual_names = list(model.coords["r2d2_split"])
            assert actual_names == expected_names


class TestR2D2Serialization:
    """Tests for R2D2 serialization."""

    def test_round_trip(self):
        """Serialization round-trip should preserve configuration."""
        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.4),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={"control": "control", "fourier": "fourier"},
        )

        # Serialize
        config = serialization.serialize(r2d2)

        # Deserialize
        restored = serialization.deserialize(config)

        # Check restoration
        assert isinstance(restored, R2D2)
        assert restored.r2 == r2d2.r2
        assert restored.total_sigma == r2d2.total_sigma
        assert restored.dims == r2d2.dims
        assert not restored.built  # Should not be built after deserialization

    def test_split_round_trip(self):
        """R2D2Split serialization round-trip."""
        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.4),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={"control": "control"},
        )

        split = r2d2.split("control")
        config = serialization.serialize(split)
        restored = serialization.deserialize(config)

        assert isinstance(restored, R2D2Split)
        assert restored.component_name == "control"
        assert restored.decomposition.dims == {"control": "control"}

    def test_sigma_round_trip(self):
        """R2D2Sigma serialization round-trip."""
        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.4),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={"control": "control"},
        )

        sigma = r2d2.error_sigma
        config = serialization.serialize(sigma)
        restored = serialization.deserialize(config)

        assert isinstance(restored, R2D2Sigma)
        assert restored.decomposition.dims == {"control": "control"}


class TestR2D2Integration:
    """Integration tests for R2D2 with MMM-like regression patterns."""

    def test_r2d2_mmm_style_regression(self):
        """R2D2 should work for MMM-style regression with media, controls."""
        # Create R2D2 decomposition - same as MMM usage
        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.4),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={
                "media": "channel",
                "control": "control",
            },
        )

        # Create dataset similar to MMM
        ds = xr.Dataset(
            {
                "media": (("obs", "channel"), np.random.randn(100, 4)),
                "controls": (("obs", "control"), np.random.randn(100, 2)),
                "y": ("obs", np.random.randn(100)),
            },
            coords={
                "obs": range(100),
                "channel": ["tv", "digital", "radio", "print"],
                "control": ["event_1", "event_2"],
            },
        )

        # Build model - this is how it would work with MMM model_config
        with pm.Model(
            coords={
                "obs": range(100),
                "channel": ["tv", "digital", "radio", "print"],
                "control": ["event_1", "event_2"],
            }
        ) as model:
            media = pmd.Data("media", ds["media"])
            controls = pmd.Data("controls", ds["controls"])

            # R2D2 splits as priors (like model_config in MMM)
            beta_media = r2d2.split("media").create_variable("beta_media")
            beta_control = r2d2.split("control").create_variable("beta_control")
            intercept = pmd.Normal("intercept", mu=0, sigma=1)

            # Linear predictor
            mu = intercept + media @ beta_media + controls @ beta_control

            # Error sigma from R2D2
            sigma = r2d2.error_sigma.create_variable("error_sigma")

            # Likelihood
            pmd.Normal("y_obs", mu=mu, sigma=sigma, observed=ds["y"])

        # Verify model structure
        assert "y_obs" in model.named_vars
        assert "beta_media" in model.named_vars
        assert "beta_control" in model.named_vars
        assert "intercept" in model.named_vars

        # Verify R2D2 tracking
        assert r2d2.built

    def test_r2d2_with_prior_likelihood(self):
        """R2D2 error_sigma should work inside Prior for likelihood."""
        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.4),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={"control": "control"},
        )

        ds = xr.Dataset(
            {
                "controls": (("obs", "control"), np.random.randn(10, 2)),
                "y": ("obs", np.ones(10)),
            },
            coords={"obs": range(10), "control": ["a", "b"]},
        )

        with pm.Model(coords={"obs": range(10), "control": ["a", "b"]}) as model:
            controls = pmd.Data("controls", ds["controls"])

            # R2D2 as VariableFactory in model_config style
            model_config = {
                "gamma_control": r2d2.split("control"),
                "likelihood_sigma": r2d2.error_sigma,
            }

            # Create variables from config
            beta = model_config["gamma_control"].create_variable("beta")
            sigma = model_config["likelihood_sigma"].create_variable("sigma")

            mu = controls @ beta
            pmd.Normal("y_obs", mu=mu, sigma=sigma, observed=ds["y"])

        # Verify it works
        assert "y_obs" in model.named_vars
        assert r2d2.built

    def test_r2d2_multiple_splits_same_decomposition(self):
        """All splits should share the same decomposition."""
        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.4),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={
                "control": "control",
            },
        )

        ds = xr.Dataset(
            {
                "controls": (("obs", "control"), np.random.randn(10, 2)),
                "y": ("obs", np.ones(10)),
            },
            coords={"obs": range(10), "control": ["a", "b"]},
        )

        with pm.Model(coords={"obs": range(10), "control": ["a", "b"]}):
            controls = pmd.Data("controls", ds["controls"])

            # Get multiple splits (same component for testing)
            split_control1 = r2d2.split("control")
            split_control2 = r2d2.split("control")
            error = r2d2.error_sigma

            # All should reference the SAME decomposition
            assert split_control1.decomposition is r2d2
            assert split_control2.decomposition is r2d2
            assert error.decomposition is r2d2

            # Create variables
            beta_control = split_control1.create_variable("beta_control")
            sigma = error.create_variable("sigma")

            mu = controls @ beta_control
            pmd.Normal("y_obs", mu=mu, sigma=sigma, observed=ds["y"])

        # Verify single decomposition built
        assert r2d2.built
        assert len(r2d2._splits) == 1  # only control

    def test_r2d2_model_config_pattern(self):
        """Test the pattern that would be used in MMM model_config."""
        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.4),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={
                "media": "channel",
                "control": "control",
            },
        )

        # This is how it would be used in MMM
        model_config = {
            "likelihood": Prior("Normal", sigma=r2d2.error_sigma),
            "gamma_control": r2d2.split("control"),
            "gamma_media": r2d2.split("media"),
        }

        # Verify config structure
        assert "likelihood" in model_config
        assert "gamma_control" in model_config
        assert "gamma_media" in model_config

        # Verify all are VariableFactory compatible
        for key, value in model_config.items():
            assert hasattr(value, "create_variable"), f"{key} missing create_variable"
            assert hasattr(value, "dims"), f"{key} missing dims attribute"


class TestR2D2WithMMM:
    """Integration tests for R2D2 with the MMM class."""

    @pytest.fixture
    def mmm_data(self):
        """Create minimal dataset for MMM."""
        import pandas as pd

        dates = pd.date_range("2020-01-01", periods=52, freq="W")
        return pd.DataFrame(
            {
                "date_week": dates,
                "x1": np.random.randn(52),
                "x2": np.random.randn(52),
                "event_1": np.random.choice([0, 1], 52),
                "event_2": np.random.choice([0, 1], 52),
                "t": range(52),
                "y": np.random.randn(52),
            }
        )

    def test_r2d2_with_mmm_build_model(self, mmm_data):
        """R2D2 should work with MMM's build_model for control components."""
        from pymc_marketing.mmm import MMM
        from pymc_marketing.mmm.components.adstock import GeometricAdstock
        from pymc_marketing.mmm.components.saturation import LogisticSaturation

        # Use R2D2 only for control components (not fourier)
        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.4),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={
                "control": "control",
            },
        )

        mmm = MMM(
            date_column="date_week",
            channel_columns=["x1", "x2"],
            control_columns=["event_1", "event_2", "t"],
            adstock=GeometricAdstock(l_max=8),
            saturation=LogisticSaturation(),
            yearly_seasonality=2,
            model_config={
                "likelihood": Prior("Normal", sigma=r2d2.error_sigma),
                "gamma_control": r2d2.split("control"),
            },
        )

        X = mmm_data.drop("y", axis=1)
        y = mmm_data["y"]

        # Build model
        mmm.build_model(X, y)

        # Verify R2D2 variables exist
        assert "r2d2_r2" in mmm.model.named_vars
        assert "r2d2_total_sigma" in mmm.model.named_vars
        assert "r2d2_weights" in mmm.model.named_vars

        # Verify R2D2 split was created
        assert "gamma_control" in mmm.model.named_vars

        # Verify the decomposition was built
        assert r2d2.built

    def test_r2d2_with_mmm_control_and_fourier_build_model(self, mmm_data):
        """R2D2 should build when sharing budget across control and Fourier."""
        from pymc_marketing.mmm import MMM
        from pymc_marketing.mmm.components.adstock import GeometricAdstock
        from pymc_marketing.mmm.components.saturation import LogisticSaturation

        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.2),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={"control": "control", "fourier": "fourier_mode"},
        )
        mmm = MMM(
            date_column="date_week",
            channel_columns=["x1", "x2"],
            control_columns=["event_1", "event_2", "t"],
            target_column="y",
            adstock=GeometricAdstock(l_max=8),
            saturation=LogisticSaturation(),
            yearly_seasonality=1,
            model_config={
                "likelihood": Prior("Normal", sigma=r2d2.error_sigma, dims="date"),
                "gamma_control": r2d2.split("control"),
                "gamma_fourier": r2d2.split("fourier"),
            },
        )

        X = mmm_data.drop("y", axis=1)
        y = mmm_data["y"]

        mmm.build_model(X, y)

        expected_split = ["event_1", "event_2", "t", "sin_1", "cos_1"]
        assert list(mmm.model.coords["r2d2_split"]) == expected_split
        free_rv_names = {v.name for v in mmm.model.free_RVs}
        assert "gamma_control" in free_rv_names
        assert "gamma_fourier" in free_rv_names
        assert mmm.model.dim_lengths["fourier_mode"].eval() == 2

    def test_r2d2_with_mmm_contribution_names(self, mmm_data):
        """R2D2 should produce variables that work with MMM contribution plotting."""
        from pymc_marketing.mmm import MMM
        from pymc_marketing.mmm.components.adstock import GeometricAdstock
        from pymc_marketing.mmm.components.saturation import LogisticSaturation

        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.4),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={
                "control": "control",
            },
        )

        mmm = MMM(
            date_column="date_week",
            channel_columns=["x1", "x2"],
            control_columns=["event_1", "event_2", "t"],
            adstock=GeometricAdstock(l_max=8),
            saturation=LogisticSaturation(),
            yearly_seasonality=2,
            model_config={
                "likelihood": Prior("Normal", sigma=r2d2.error_sigma),
                "gamma_control": r2d2.split("control"),
            },
        )

        X = mmm_data.drop("y", axis=1)
        y = mmm_data["y"]

        # Build model
        mmm.build_model(X, y)

        # Verify contribution deterministics exist (needed for plotting)
        assert "control_contribution" in mmm.model.named_vars
        assert "yearly_seasonality_contribution" in mmm.model.named_vars

        # Verify all expected variables
        expected_vars = [
            "r2d2_r2",
            "r2d2_total_sigma",
            "r2d2_weights",
            "gamma_control",
            "intercept_contribution",
            "control_contribution",
            "yearly_seasonality_contribution",
        ]
        for var in expected_vars:
            assert var in mmm.model.named_vars, f"Missing variable: {var}"

    def test_r2d2_mmm_save_load_round_trip(self, mmm_data, tmp_path, mock_pymc_sample):
        """MMM with R2D2 should survive save/load with shared decomposition preserved."""
        from pymc_marketing.mmm import MMM
        from pymc_marketing.mmm.components.adstock import GeometricAdstock
        from pymc_marketing.mmm.components.saturation import LogisticSaturation

        r2d2 = R2D2(
            r2=Prior("Beta", mu=0.8, sigma=0.2),
            total_sigma=Prior("LogNormal", mu=0, sigma=1),
            dims={
                "control": "control",
            },
        )

        mmm = MMM(
            date_column="date_week",
            channel_columns=["x1", "x2"],
            control_columns=["event_1", "event_2", "t"],
            adstock=GeometricAdstock(l_max=8),
            saturation=LogisticSaturation(),
            yearly_seasonality=2,
            model_config={
                "likelihood": Prior("Normal", sigma=r2d2.error_sigma),
                "gamma_control": r2d2.split("control"),
            },
        )

        X = mmm_data.drop("y", axis=1)
        y = mmm_data["y"]

        mmm.fit(X, y, chains=1, draws=10, tune=10, random_seed=42)

        original_id = mmm.id

        save_path = tmp_path / "mmm_r2d2.nc"
        mmm.save(str(save_path))

        loaded = MMM.load(str(save_path), check=True)

        assert loaded.id == original_id

        gamma_control = loaded.model_config["gamma_control"]
        likelihood = loaded.model_config["likelihood"]
        sigma = likelihood.parameters["sigma"]
        assert gamma_control.decomposition is sigma.decomposition, (
            "Shared decomposition reference should be preserved after load"
        )

        assert "r2d2_weights" in loaded.model.named_vars
