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
Unit tests for the Pydantic YAML configuration schema models.

These tests validate the schema models in isolation (no model building).
"""

import pytest
import yaml
from pydantic import ValidationError

from pymc_marketing.mmm.builders.schema import (
    BuildSpec,
    CalibrationStep,
    DataConfig,
    MMMYamlConfig,
    suggest_typo_fix,
)


class TestSuggestTypoFix:
    def test_close_match(self):
        assert suggest_typo_fix("modle", {"model", "data"}) == "model"

    def test_no_match(self):
        assert suggest_typo_fix("zzz", {"model", "data"}) is None

    def test_close_match_kwargs(self):
        assert suggest_typo_fix("kwarg", {"class", "kwargs", "args"}) == "kwargs"

    def test_close_match_effects(self):
        valid = {"model", "data", "effects", "original_scale_vars"}
        assert suggest_typo_fix("efects", valid) == "effects"


class TestBuildSpec:
    def test_valid_minimal(self):
        spec = BuildSpec.model_validate({"class": "foo.Bar"})
        assert spec.class_ == "foo.Bar"
        assert spec.kwargs == {}
        assert spec.args == []

    def test_valid_full(self):
        spec = BuildSpec.model_validate(
            {"class": "foo.Bar", "kwargs": {"x": 1}, "args": [2]}
        )
        assert spec.class_ == "foo.Bar"
        assert spec.kwargs == {"x": 1}
        assert spec.args == [2]

    def test_missing_class_raises(self):
        with pytest.raises(ValidationError):
            BuildSpec.model_validate({"kwargs": {"x": 1}})

    def test_non_string_class_raises(self):
        with pytest.raises(ValidationError):
            BuildSpec.model_validate({"class": 123})

    def test_unknown_key_with_suggestion(self):
        with pytest.raises(ValidationError, match="Did you mean 'kwargs'"):
            BuildSpec.model_validate({"class": "foo.Bar", "kwarg": {}})

    def test_unknown_key_without_suggestion(self):
        with pytest.raises(ValidationError, match="Unknown key 'zzz'"):
            BuildSpec.model_validate({"class": "foo.Bar", "zzz": {}})

    def test_model_dump_round_trip(self):
        original = {"class": "foo.Bar", "kwargs": {"x": 1}}
        spec = BuildSpec.model_validate(original)
        dumped = spec.model_dump(by_alias=True)
        assert dumped["class"] == "foo.Bar"
        assert dumped["kwargs"] == {"x": 1}
        assert dumped["args"] == []

    def test_nested_build_specs_in_kwargs_pass_through(self):
        """Nested build specs inside kwargs are not recursively validated."""
        raw = {
            "class": "foo.Bar",
            "kwargs": {
                "nested": {
                    "class": "baz.Qux",
                    "kwargs": {"y": 2},
                }
            },
        }
        spec = BuildSpec.model_validate(raw)
        assert spec.kwargs["nested"]["class"] == "baz.Qux"

    def test_populate_by_name(self):
        """class_ can be set by Python name (not just alias)."""
        spec = BuildSpec(class_="foo.Bar")
        assert spec.class_ == "foo.Bar"


class TestDataConfig:
    def test_valid(self):
        cfg = DataConfig.model_validate(
            {"X_path": "data/X.csv", "y_path": "data/y.csv"}
        )
        assert cfg.X_path == "data/X.csv"
        assert cfg.y_path == "data/y.csv"

    def test_empty(self):
        cfg = DataConfig.model_validate({})
        assert cfg.X_path is None
        assert cfg.y_path is None

    def test_unknown_key_raises(self):
        with pytest.raises(ValidationError):
            DataConfig.model_validate({"X_path": "a.csv", "z_path": "b.csv"})


class TestCalibrationStep:
    def test_valid_with_params(self):
        step = CalibrationStep.model_validate(
            {"add_lift_test_measurements": {"df": "some_value"}}
        )
        assert step.method_name == "add_lift_test_measurements"
        assert step.params == {"df": "some_value"}

    def test_valid_with_none_params(self):
        step = CalibrationStep.model_validate({"some_method": None})
        assert step.method_name == "some_method"
        assert step.params is None

    def test_empty_dict_raises(self):
        with pytest.raises(ValidationError, match="exactly one method"):
            CalibrationStep.model_validate({})

    def test_multi_key_dict_raises(self):
        with pytest.raises(ValidationError, match="exactly one method"):
            CalibrationStep.model_validate({"method_a": {}, "method_b": {}})

    def test_non_dict_raises(self):
        with pytest.raises(ValidationError, match="exactly one method"):
            CalibrationStep.model_validate("not-a-dict")

    def test_extra_keys_alongside_method_name_raises(self):
        with pytest.raises(ValidationError):
            CalibrationStep.model_validate(
                {"method_name": "foo", "params": {"k": 1}, "extra": "bad"}
            )

    def test_direct_construction(self):
        step = CalibrationStep(method_name="foo", params={"k": 1})
        assert step.method_name == "foo"
        assert step.params == {"k": 1}

    def test_model_dump_round_trip(self):
        step = CalibrationStep.model_validate(
            {"add_lift_test_measurements": {"df": "some_value"}}
        )
        dumped = step.model_dump()
        restored = CalibrationStep.model_validate(dumped)
        assert restored.method_name == step.method_name
        assert restored.params == step.params


class TestMMMYamlConfig:
    def test_minimal_valid(self):
        cfg = MMMYamlConfig.model_validate({"model": {"class": "some.Class"}})
        assert cfg.model.class_ == "some.Class"
        assert cfg.data is None
        assert cfg.effects is None
        assert cfg.original_scale_vars is None
        assert cfg.calibration is None
        assert cfg.idata_path is None

    def test_full_valid(self):
        raw = {
            "model": {"class": "some.Class", "kwargs": {"x": 1}},
            "data": {"X_path": "data/X.csv", "y_path": "data/y.csv"},
            "effects": [{"class": "some.Effect", "kwargs": {}}],
            "original_scale_vars": ["channel_contribution"],
            "calibration": [{"method_a": {"param": 1}}],
            "idata_path": "data/idata.nc",
        }
        cfg = MMMYamlConfig.model_validate(raw)
        assert cfg.model.class_ == "some.Class"
        assert cfg.data is not None
        assert cfg.data.X_path == "data/X.csv"
        assert len(cfg.effects) == 1
        assert cfg.original_scale_vars == ["channel_contribution"]
        assert len(cfg.calibration) == 1
        assert cfg.calibration[0].method_name == "method_a"
        assert cfg.idata_path == "data/idata.nc"

    def test_missing_model_raises(self):
        with pytest.raises(ValidationError):
            MMMYamlConfig.model_validate({"effects": []})

    def test_typo_model(self):
        with pytest.raises(ValidationError, match="Did you mean 'model'"):
            MMMYamlConfig.model_validate({"modle": {"class": "some.Class"}})

    def test_typo_effects(self):
        with pytest.raises(ValidationError, match="Did you mean 'effects'"):
            MMMYamlConfig.model_validate(
                {
                    "model": {"class": "some.Class"},
                    "efects": [],
                }
            )

    def test_typo_calibration(self):
        with pytest.raises(ValidationError, match="Did you mean 'calibration'"):
            MMMYamlConfig.model_validate(
                {
                    "model": {"class": "some.Class"},
                    "calibraton": [],
                }
            )

    def test_unknown_key_no_suggestion(self):
        with pytest.raises(ValidationError, match="Unknown config key 'zzz'"):
            MMMYamlConfig.model_validate(
                {
                    "model": {"class": "some.Class"},
                    "zzz": "value",
                }
            )

    def test_original_scale_vars_none_accepted(self):
        cfg = MMMYamlConfig.model_validate(
            {
                "model": {"class": "some.Class"},
                "original_scale_vars": None,
            }
        )
        assert cfg.original_scale_vars is None

    def test_calibration_none_accepted(self):
        cfg = MMMYamlConfig.model_validate(
            {
                "model": {"class": "some.Class"},
                "calibration": None,
            }
        )
        assert cfg.calibration is None

    def test_model_dump_round_trip_with_calibration(self):
        raw = {
            "model": {"class": "some.Class", "kwargs": {"x": 1}},
            "calibration": [{"method_a": {"param": 1}}],
        }
        cfg = MMMYamlConfig.model_validate(raw)
        dumped = cfg.model_dump(by_alias=True)
        restored = MMMYamlConfig.model_validate(dumped)
        assert restored.model.class_ == cfg.model.class_
        assert len(restored.calibration) == 1
        assert restored.calibration[0].method_name == "method_a"
        assert restored.calibration[0].params == {"param": 1}


class TestFromYamlFile:
    def test_valid_file(self, tmp_path):
        config = {
            "model": {
                "class": "pymc_marketing.mmm.multidimensional.MMM",
                "kwargs": {"date_column": "date"},
            }
        }
        path = tmp_path / "config.yml"
        path.write_text(yaml.dump(config))

        cfg = MMMYamlConfig.from_yaml_file(path)
        assert cfg.model.class_ == "pymc_marketing.mmm.multidimensional.MMM"

    def test_invalid_file_raises_validation_error(self, tmp_path):
        config = {"modle": {"class": "some.Class"}}
        path = tmp_path / "bad.yml"
        path.write_text(yaml.dump(config))

        with pytest.raises(ValidationError, match="Did you mean 'model'"):
            MMMYamlConfig.from_yaml_file(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MMMYamlConfig.from_yaml_file(tmp_path / "nonexistent.yml")
