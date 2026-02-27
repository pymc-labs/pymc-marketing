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
"""
Tests for the YAML builder module in pymc_marketing.mmm.builders.yaml.
"""

from pathlib import Path

import pandas as pd
import pytest
import xarray as xr
import yaml

from pymc_marketing.mmm.builders.yaml import (
    _apply_and_validate_calibration_steps,
    build_mmm_from_yaml,
)
from pymc_marketing.model_config import ModelConfigError


@pytest.fixture
def X_data():
    """Load X data for testing."""
    return pd.read_csv("data/processed/X.csv")


@pytest.fixture
def y_data():
    """Load y data for testing."""
    return pd.read_csv("data/processed/y.csv")


def get_yaml_files():
    """Get all YAML files from the data/config_files directory."""
    config_dir = Path("data/config_files")
    return [
        str(file)
        for file in config_dir.glob("*.yml")
        if "wrong_" not in file.name
        and "multi_dimensional_example_model.yml" not in file.name
        and "multi_dimensional_fivetran.yml" not in file.name
    ]


@pytest.mark.parametrize(
    "model_kwargs",
    [
        None,
        {
            "adstock": {
                "class": "pymc_marketing.mmm.components.adstock.GeometricAdstock",
                "kwargs": {"l_max": 28},
            }
        },
        {"time_varying_intercept": False},
    ],
)
@pytest.mark.parametrize("config_path", get_yaml_files())
def test_build_mmm_from_yaml(config_path, X_data, y_data, model_kwargs):
    """Test that build_mmm_from_yaml can create models from all config files."""
    # Load YAML to check if effects are defined
    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Build model from YAML
    model = build_mmm_from_yaml(
        config_path=config_path, X=X_data, y=y_data.squeeze(), model_kwargs=model_kwargs
    )

    # Check that model was created successfully
    assert model is not None

    # Check that model has the expected structure
    assert hasattr(model, "model")

    # Only check for mu_effects if effects are defined in the YAML
    if config.get("effects"):
        assert hasattr(model, "mu_effects")
        assert len(model.mu_effects) > 0

    # Test that prior predictive sampling works using the model's method
    prior_predictive = model.sample_prior_predictive(X=X_data, y=y_data, samples=10)

    # Verify we have a prior predictive result
    assert prior_predictive is not None

    # Verify that the result is an xarray dataset
    assert isinstance(prior_predictive, xr.Dataset)

    if model_kwargs:
        # assert that model_kwargs are reflected in the model
        for key, value in model_kwargs.items():
            attr = getattr(model, key, None)
            if isinstance(value, dict) and "class" in value and "kwargs" in value:
                # Check class name
                expected_class_name = value["class"].split(".")[-1]
                assert attr.__class__.__name__ == expected_class_name
                # Check only the specified kwargs
                for k, v in value["kwargs"].items():
                    assert hasattr(attr, k), f"{key} missing attribute {k}"
                    assert getattr(attr, k) == v, f"{key}.{k}={getattr(attr, k)} != {v}"
            else:
                assert attr == value


def test_wrong_adstock_class():
    """Test that a model with a wrong adstock class fails appropriately."""
    wrong_config_path = Path("tests/mmm/builders/config_files/wrong_adstock_class.yml")

    # Should fail with AttributeError for the non-existent adstock class
    with pytest.raises(AttributeError, match=r".*NonExistentAdstock.*"):
        build_mmm_from_yaml(wrong_config_path)

    # Verify the config file has the expected wrong class
    cfg = yaml.safe_load(wrong_config_path.read_text())
    adstock_config = cfg["model"]["kwargs"]["adstock"]
    assert adstock_config["class"] == "pymc_marketing.mmm.NonExistentAdstock"


def test_wrong_saturation_params():
    """Test that a model with wrong saturation parameters fails appropriately."""
    wrong_config_path = Path(
        "tests/mmm/builders/config_files/wrong_saturation_params.yml"
    )

    # Should eventually fail with a ModelConfigError or TypeError
    with pytest.raises((TypeError, ModelConfigError, ValueError)):
        build_mmm_from_yaml(wrong_config_path)

    # Verify the config file has the expected wrong parameters
    cfg = yaml.safe_load(wrong_config_path.read_text())
    saturation_config = cfg["model"]["kwargs"]["saturation"]["kwargs"]["priors"]
    assert saturation_config["alpha"] == "not_a_number"
    assert saturation_config["lambda"] == -5.0


def test_wrong_distribution():
    """Test that a model with an invalid distribution fails appropriately."""
    wrong_config_path = Path("tests/mmm/builders/config_files/wrong_distribution.yml")

    # Should fail with ModelConfigError when parsing distributions
    with pytest.raises(ModelConfigError):
        build_mmm_from_yaml(wrong_config_path)

    # Verify the config file has the expected wrong distribution
    cfg = yaml.safe_load(wrong_config_path.read_text())
    model_config = cfg["model"]["kwargs"]["model_config"]
    assert model_config["intercept"]["dist"] == "InvalidDistribution"


def test_wrong_parameter_type():
    """Test that a model with a wrong parameter type fails appropriately."""
    wrong_config_path = Path("tests/mmm/builders/config_files/wrong_parameter_type.yml")

    # Should fail with ModelConfigError when parsing distributions
    with pytest.raises(ModelConfigError):
        build_mmm_from_yaml(wrong_config_path)

    # Verify the config file has the expected wrong parameter type
    cfg = yaml.safe_load(wrong_config_path.read_text())
    model_config = cfg["model"]["kwargs"]["model_config"]
    assert model_config["likelihood"]["kwargs"]["sigma"] == "wrong_value_type"


@pytest.fixture
def dummy_mmm():
    class DummyMMM:
        def __init__(self) -> None:
            self.called_with = None

        def good(self, **kwargs) -> None:
            self.called_with = kwargs

        def add_lift_test_measurements(self, **kwargs) -> None:
            self.called_with = kwargs

    return DummyMMM()


@pytest.fixture
def failing_mmm(dummy_mmm):
    class FailingMMM(type(dummy_mmm)):
        def failing(self, **kwargs) -> None:  # type: ignore[override]
            raise ValueError("boom")

    failing = FailingMMM()
    failing.good = dummy_mmm.good  # type: ignore[attr-defined]
    failing.add_lift_test_measurements = dummy_mmm.add_lift_test_measurements  # type: ignore[attr-defined]
    return failing


def test_apply_and_validate_calibration_steps_success(dummy_mmm, tmp_path):
    cfg = {
        "calibration": [
            {
                "good": {
                    "df": {
                        "class": "pandas.DataFrame",
                        "kwargs": {"data": {"a": [1]}},
                    }
                }
            }
        ]
    }

    _apply_and_validate_calibration_steps(dummy_mmm, cfg, tmp_path)

    assert isinstance(dummy_mmm.called_with["df"], pd.DataFrame)


def test_apply_calibration_non_list(dummy_mmm, tmp_path):
    cfg = {"calibration": "not-a-list"}

    with pytest.raises(TypeError, match="must be a list"):
        _apply_and_validate_calibration_steps(dummy_mmm, cfg, tmp_path)


def test_apply_calibration_step_not_mapping(dummy_mmm, tmp_path):
    cfg = {"calibration": ["not-mapping"]}

    with pytest.raises(TypeError, match="must be a mapping"):
        _apply_and_validate_calibration_steps(dummy_mmm, cfg, tmp_path)


def test_apply_calibration_step_multiple_methods(dummy_mmm, tmp_path):
    cfg = {"calibration": [{"good": {}, "other": {}}]}

    with pytest.raises(ValueError, match="single method"):
        _apply_and_validate_calibration_steps(dummy_mmm, cfg, tmp_path)


def test_apply_calibration_missing_method(dummy_mmm, tmp_path):
    cfg = {"calibration": [{"missing": {}}]}

    with pytest.raises(AttributeError, match="has no calibration method 'missing'"):
        _apply_and_validate_calibration_steps(dummy_mmm, cfg, tmp_path)


def test_apply_calibration_not_callable(dummy_mmm, tmp_path):
    dummy_mmm.not_callable = 123  # type: ignore[attr-defined]
    cfg = {"calibration": [{"not_callable": {}}]}

    with pytest.raises(TypeError, match="not callable"):
        _apply_and_validate_calibration_steps(dummy_mmm, cfg, tmp_path)


def test_apply_calibration_params_not_mapping(dummy_mmm, tmp_path):
    cfg = {"calibration": [{"good": "invalid"}]}

    with pytest.raises(TypeError, match="must be a mapping"):
        _apply_and_validate_calibration_steps(dummy_mmm, cfg, tmp_path)


def test_apply_calibration_dist_disallowed(dummy_mmm, tmp_path):
    cfg = {"calibration": [{"add_lift_test_measurements": {"dist": "Gamma"}}]}

    with pytest.raises(ValueError, match="`dist` parameter"):
        _apply_and_validate_calibration_steps(dummy_mmm, cfg, tmp_path)


def test_apply_calibration_propagates_failure(failing_mmm, tmp_path):
    cfg = {"calibration": [{"failing": {}}]}

    with pytest.raises(
        RuntimeError, match="Failed to apply calibration step 'failing'"
    ):
        _apply_and_validate_calibration_steps(failing_mmm, cfg, tmp_path)


def test_special_prior_in_yaml(tmp_path, mock_pymc_sample):
    """Test that SpecialPrior (LogNormalPrior) works in YAML config."""
    # Create test data
    X = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=100),
            "channel_1": range(100),
            "channel_2": range(100, 200),
        }
    )
    y = pd.Series(range(100), name="y")

    # Create YAML config with LogNormalPrior
    config = {
        "model": {
            "class": "pymc_marketing.mmm.multidimensional.MMM",
            "kwargs": {
                "date_column": "date",
                "channel_columns": ["channel_1", "channel_2"],
                "target_column": "y",
                "adstock": {
                    "class": "pymc_marketing.mmm.GeometricAdstock",
                    "kwargs": {"l_max": 4},
                },
                "saturation": {
                    "class": "pymc_marketing.mmm.LogisticSaturation",
                    "kwargs": {
                        "priors": {
                            "lam": {
                                "special_prior": "LogNormalPrior",
                                "mean": 1.0,
                                "std": 0.5,
                                "dims": ["channel"],
                            },
                            "beta": {
                                "distribution": "HalfNormal",
                                "sigma": 1.0,
                                "dims": ["channel"],
                            },
                        }
                    },
                },
                "sampler_config": {
                    "draws": 10,
                    "tune": 10,
                    "chains": 1,
                    "random_seed": 42,
                },
            },
        }
    }

    # Write to temp file
    config_path = tmp_path / "test_config.yml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Build model from YAML
    model = build_mmm_from_yaml(config_path, X=X, y=y)

    # Check that model was created successfully
    assert model is not None
    assert hasattr(model, "saturation")

    # Check that the prior was deserialized correctly
    assert "lam" in model.saturation.priors
    lam_prior = model.saturation.priors["lam"]

    # Check it has the special_prior attribute and correct values
    assert hasattr(lam_prior, "to_dict")
    lam_dict = lam_prior.to_dict()
    assert lam_dict.get("special_prior") == "LogNormalPrior"
    # Parameters are stored in kwargs subdictionary
    assert lam_dict.get("kwargs", {}).get("mean") == 1.0
    assert lam_dict.get("kwargs", {}).get("std") == 0.5
    assert lam_dict.get("dims") == ("channel",)

    # Fit the model to ensure SpecialPrior works end-to-end
    model.fit(X=X, y=y)

    # Check that the model has inference data after fitting
    assert model.idata is not None
    assert "posterior" in model.idata
