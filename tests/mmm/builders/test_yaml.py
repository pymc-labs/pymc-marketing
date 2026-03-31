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
from pydantic import ValidationError

from pymc_marketing.mmm.builders.schema import CalibrationStep, MMMYamlConfig
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


@pytest.fixture
def _minimal_model_config():
    """Reusable minimal MMM config dict (no data/idata sections)."""
    return {
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
                    "kwargs": {},
                },
            },
        }
    }


@pytest.fixture
def _sample_data():
    """Small X / y pair for lightweight integration tests."""
    X = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=52, freq="W"),
            "channel_1": range(52),
            "channel_2": range(100, 152),
        }
    )
    y = pd.Series(range(52), name="y")
    return X, y


def get_yaml_files():
    """Get all YAML files from the data/config_files directory."""
    config_dir = Path("data/config_files")
    return [
        str(file)
        for file in config_dir.glob("*.yml")
        if "wrong_" not in file.name
        and "multi_dimensional_example_model.yml" not in file.name
        and "multi_dimensional_fivetran.yml" not in file.name
        # Exclude cost_per_unit example: it's a documentation-focused config
        # that exercises cost-per-unit MMM behavior separately from the
        # generic YAML builder smoke tests in this module.
        and "cost_per_unit_example.yml" not in file.name
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
    with open(config_path) as file:
        config = yaml.safe_load(file)

    model = build_mmm_from_yaml(
        config_path=config_path,
        X=X_data,
        y=y_data.squeeze(),
        model_kwargs=model_kwargs,
    )

    assert model is not None
    assert hasattr(model, "model")

    if config.get("effects"):
        assert hasattr(model, "mu_effects")
        assert len(model.mu_effects) > 0

    prior_predictive = model.sample_prior_predictive(X=X_data, y=y_data, samples=10)
    assert prior_predictive is not None
    assert isinstance(prior_predictive, xr.Dataset)

    if model_kwargs:
        for key, value in model_kwargs.items():
            attr = getattr(model, key, None)
            if isinstance(value, dict) and "class" in value and "kwargs" in value:
                expected_class_name = value["class"].split(".")[-1]
                assert attr.__class__.__name__ == expected_class_name
                for k, v in value["kwargs"].items():
                    assert hasattr(attr, k), f"{key} missing attribute {k}"
                    assert getattr(attr, k) == v, f"{key}.{k}={getattr(attr, k)} != {v}"
            else:
                assert attr == value


def test_wrong_adstock_class():
    """Test that a model with a wrong adstock class fails appropriately."""
    wrong_config_path = Path("tests/mmm/builders/config_files/wrong_adstock_class.yml")

    with pytest.raises(AttributeError, match=r".*NonExistentAdstock.*"):
        build_mmm_from_yaml(wrong_config_path)

    cfg = yaml.safe_load(wrong_config_path.read_text())
    adstock_config = cfg["model"]["kwargs"]["adstock"]
    assert adstock_config["class"] == "pymc_marketing.mmm.NonExistentAdstock"


def test_wrong_saturation_params():
    """Test that a model with wrong saturation parameters fails appropriately."""
    wrong_config_path = Path(
        "tests/mmm/builders/config_files/wrong_saturation_params.yml"
    )

    with pytest.raises((TypeError, ModelConfigError, ValueError)):
        build_mmm_from_yaml(wrong_config_path)

    cfg = yaml.safe_load(wrong_config_path.read_text())
    saturation_config = cfg["model"]["kwargs"]["saturation"]["kwargs"]["priors"]
    assert saturation_config["alpha"] == "not_a_number"
    assert saturation_config["lambda"] == -5.0


def test_wrong_distribution():
    """Test that a model with an invalid distribution fails appropriately."""
    wrong_config_path = Path("tests/mmm/builders/config_files/wrong_distribution.yml")

    with pytest.raises(ModelConfigError):
        build_mmm_from_yaml(wrong_config_path)

    cfg = yaml.safe_load(wrong_config_path.read_text())
    model_config = cfg["model"]["kwargs"]["model_config"]
    assert model_config["intercept"]["dist"] == "InvalidDistribution"


def test_wrong_parameter_type():
    """Test that a model with a wrong parameter type fails appropriately."""
    wrong_config_path = Path("tests/mmm/builders/config_files/wrong_parameter_type.yml")

    with pytest.raises(ModelConfigError):
        build_mmm_from_yaml(wrong_config_path)

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


def test_apply_and_validate_calibration_steps_success(dummy_mmm):
    steps = [
        CalibrationStep.model_validate(
            {
                "good": {
                    "df": {
                        "class": "pandas.DataFrame",
                        "kwargs": {"data": {"a": [1]}},
                    }
                }
            }
        )
    ]

    _apply_and_validate_calibration_steps(dummy_mmm, steps)

    assert isinstance(dummy_mmm.called_with["df"], pd.DataFrame)


def test_apply_calibration_missing_method(dummy_mmm):
    steps = [CalibrationStep.model_validate({"missing": {}})]

    with pytest.raises(AttributeError, match="has no calibration method 'missing'"):
        _apply_and_validate_calibration_steps(dummy_mmm, steps)


def test_apply_calibration_not_callable(dummy_mmm):
    dummy_mmm.not_callable = 123  # type: ignore[attr-defined]
    steps = [CalibrationStep.model_validate({"not_callable": {}})]

    with pytest.raises(TypeError, match="not callable"):
        _apply_and_validate_calibration_steps(dummy_mmm, steps)


def test_apply_calibration_dist_disallowed(dummy_mmm):
    steps = [
        CalibrationStep.model_validate(
            {"add_lift_test_measurements": {"dist": "Gamma"}}
        )
    ]

    with pytest.raises(ValueError, match="`dist` parameter"):
        _apply_and_validate_calibration_steps(dummy_mmm, steps)


def test_apply_calibration_propagates_failure(failing_mmm):
    steps = [CalibrationStep.model_validate({"failing": {}})]

    with pytest.raises(
        RuntimeError, match="Failed to apply calibration step 'failing'"
    ):
        _apply_and_validate_calibration_steps(failing_mmm, steps)


def test_special_prior_in_yaml(tmp_path, mock_pymc_sample):
    """Test that SpecialPrior (LogNormalPrior) works in YAML config."""
    X = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=100),
            "channel_1": range(100),
            "channel_2": range(100, 200),
        }
    )
    y = pd.Series(range(100), name="y")

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

    config_path = tmp_path / "test_config.yml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    model = build_mmm_from_yaml(config_path, X=X, y=y)

    assert model is not None
    assert hasattr(model, "saturation")

    assert "lam" in model.saturation.priors
    lam_prior = model.saturation.priors["lam"]

    assert hasattr(lam_prior, "to_dict")
    lam_dict = lam_prior.to_dict()
    assert lam_dict.get("special_prior") == "LogNormalPrior"
    assert lam_dict.get("kwargs", {}).get("mean") == 1.0
    assert lam_dict.get("kwargs", {}).get("std") == 0.5
    assert lam_dict.get("dims") == ("channel",)

    model.fit(X=X, y=y)

    assert model.idata is not None
    assert "posterior" in model.idata


def test_build_mmm_loads_data_from_yaml_paths(
    tmp_path, _minimal_model_config, _sample_data
):
    """X and y loaded from CSV paths declared in ``data`` section."""
    X, y = _sample_data
    X.to_csv(tmp_path / "X.csv", index=False)
    y.to_csv(tmp_path / "y.csv", index=False)

    config = _minimal_model_config
    config["data"] = {
        "X_path": str(tmp_path / "X.csv"),
        "y_path": str(tmp_path / "y.csv"),
    }

    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.dump(config))

    model = build_mmm_from_yaml(config_path)

    assert model is not None
    assert hasattr(model, "model")


def test_build_mmm_raises_when_X_missing_and_no_data_path(
    tmp_path, _minimal_model_config
):
    """ValueError when X is not passed and YAML has no ``data.X_path``."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.dump(_minimal_model_config))

    with pytest.raises(ValueError, match="X not provided"):
        build_mmm_from_yaml(config_path)


def test_build_mmm_raises_when_y_missing_and_no_data_path(
    tmp_path, _minimal_model_config, _sample_data
):
    """ValueError when y is not passed and YAML has no ``data.y_path``."""
    X, _ = _sample_data

    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.dump(_minimal_model_config))

    with pytest.raises(ValueError, match="y not provided"):
        build_mmm_from_yaml(config_path, X=X)


def test_build_mmm_loads_idata_from_path(tmp_path, _minimal_model_config, _sample_data):
    """idata_path in YAML causes InferenceData to be loaded into model."""
    import arviz as az

    X, y = _sample_data

    idata = az.from_dict(posterior={"intercept": [1.0, 2.0, 3.0]})
    idata_file = tmp_path / "idata.nc"
    idata.to_netcdf(str(idata_file))

    config = _minimal_model_config
    config["idata_path"] = str(idata_file)

    config_path = tmp_path / "config.yml"
    config_path.write_text(yaml.dump(config))

    model = build_mmm_from_yaml(config_path, X=X, y=y)

    assert model.idata is not None
    assert "posterior" in model.idata


class TestValidateConfigStructure:
    """Tests for MMMYamlConfig detecting mis-indented/invalid YAML keys."""

    def test_valid_config(self):
        """No error for a well-formed config."""
        cfg = {
            "model": {"class": "some.Class", "kwargs": {}},
            "original_scale_vars": ["channel_contribution"],
            "calibration": [{"method_a": {"param": 1}}],
            "effects": [{"class": "some.Effect", "kwargs": {}}],
        }
        MMMYamlConfig.model_validate(cfg)

    def test_missing_model_key(self):
        with pytest.raises(ValidationError):
            MMMYamlConfig.model_validate({"effects": []})

    def test_model_not_a_mapping(self):
        with pytest.raises(ValidationError):
            MMMYamlConfig.model_validate({"model": "not-a-dict"})

    @pytest.mark.parametrize(
        "misplaced_key",
        [
            "original_scale_vars",
            "calibration",
            "effects",
            "data",
            "idata_path",
        ],
    )
    def test_misplaced_key_under_model(self, misplaced_key):
        cfg = {
            "model": {
                "class": "some.Class",
                "kwargs": {},
                misplaced_key: "value",
            }
        }
        with pytest.raises(ValidationError):
            MMMYamlConfig.model_validate(cfg)


@pytest.mark.parametrize(
    "config_path",
    [
        "tests/mmm/builders/config_files/wrong_nested_original_scale_vars.yml",
        "tests/mmm/builders/config_files/wrong_nested_calibration.yml",
        "tests/mmm/builders/config_files/wrong_nested_effects.yml",
    ],
)
def test_build_mmm_rejects_nested_keys(config_path, X_data, y_data):
    """YAML files with mis-indented top-level keys should raise ValidationError."""
    with pytest.raises(ValidationError):
        build_mmm_from_yaml(config_path, X=X_data, y=y_data.squeeze())


def test_original_scale_vars_none_is_harmless(tmp_path):
    """original_scale_vars: (None in YAML) should not crash or skip silently."""
    X = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=52, freq="W"),
            "channel_1": range(52),
            "channel_2": range(100, 152),
        }
    )
    y = pd.Series(range(52), name="y")

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
                    "kwargs": {},
                },
            },
        },
        "original_scale_vars": None,
    }

    config_path = tmp_path / "config.yml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    model = build_mmm_from_yaml(config_path, X=X, y=y)
    det_names = [v.name for v in model.model.deterministics]
    assert "channel_contribution_original_scale" not in det_names


def test_calibration_none_is_harmless(dummy_mmm):
    """calibration: None should not crash."""
    _apply_and_validate_calibration_steps(dummy_mmm, None)
    assert dummy_mmm.called_with is None


def test_original_scale_vars_ordering_does_not_matter(X_data, y_data):
    """The at-end config must create original_scale variables and apply lift test."""
    config_path = "data/config_files/original_scale_vars_at_end.yml"

    model = build_mmm_from_yaml(config_path, X=X_data, y=y_data.squeeze())

    det_names = [v.name for v in model.model.deterministics]
    assert "channel_contribution_original_scale" in det_names
    assert "intercept_contribution_original_scale" in det_names

    obs_names = [rv.name for rv in model.model.observed_RVs]
    assert any("lift" in name for name in obs_names)
