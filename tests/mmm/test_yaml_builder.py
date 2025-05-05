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

from pathlib import Path

import pandas as pd
import pytest
import xarray as xr
import yaml

from pymc_marketing.mmm.builders.yaml import build_from_yaml


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
    return [str(file) for file in config_dir.glob("*.yml")]


@pytest.mark.parametrize("config_path", get_yaml_files())
def test_build_from_yaml(config_path, X_data, y_data):
    """Test that build_from_yaml can create models from all config files."""
    # Load YAML to check if effects are defined
    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Build model from YAML
    model = build_from_yaml(
        config_path=config_path,
        X=X_data,
        y=y_data,
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
