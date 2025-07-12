#   Copyright 2022 - 2025 The PyMC Labs Developers
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

import warnings
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr
import yaml

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    from pymc_marketing.mmm.builders.yaml import build_mmm_from_yaml

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
    return [str(file) for file in config_dir.glob("*.yml") if "wrong_" not in file.name]


@pytest.mark.parametrize("config_path", get_yaml_files())
def test_build_mmm_from_yaml(config_path, X_data, y_data):
    """Test that build_mmm_from_yaml can create models from all config files."""
    # Load YAML to check if effects are defined
    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Build model from YAML
    model = build_mmm_from_yaml(
        config_path=config_path,
        X=X_data,
        y=y_data.squeeze(),
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


def test_wrong_adstock_class():
    """Test that a model with a wrong adstock class fails appropriately."""
    wrong_config_path = Path("tests/mmm/builders/config_files/wrong_adstock_class.yml")

    # Should fail with AttributeError for the non-existent adstock class
    with pytest.raises(AttributeError, match=".*NonExistentAdstock.*"):
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
