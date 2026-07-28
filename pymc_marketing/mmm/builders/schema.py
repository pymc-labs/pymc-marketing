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
"""Pydantic schemas for validating MMM YAML configuration files."""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

import yaml  # type: ignore
from pydantic import BaseModel, ConfigDict, Field, model_validator


def suggest_typo_fix(
    key: str, valid_keys: set[str], *, cutoff: float = 0.6
) -> str | None:
    """Return the closest match for *key* from *valid_keys*, or ``None``.

    Parameters
    ----------
    key : str
        The unknown key that was provided.
    valid_keys : set of str
        The set of valid key names.
    cutoff : float, default 0.6
        Minimum similarity ratio for ``difflib.get_close_matches``.

    Returns
    -------
    str or None
        The closest matching key, or ``None`` if no close match is found.
    """
    matches = difflib.get_close_matches(key, valid_keys, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def _valid_field_keys(cls: type[BaseModel]) -> set[str]:
    """Derive all accepted input keys from a Pydantic model's field definitions.

    Collects both the Python field name and the alias (if any) for every
    field declared on *cls*, so that YAML keys using either form are
    recognised as valid.

    Parameters
    ----------
    cls : type of BaseModel
        The Pydantic model class to inspect.

    Returns
    -------
    set of str
        Union of field names and their aliases.
    """
    keys: set[str] = set()
    for name, field_info in cls.model_fields.items():
        keys.add(name)
        if field_info.alias:
            keys.add(field_info.alias)
    return keys


def _check_for_unknown_keys(cls: type[BaseModel], data: Any, *, label: str) -> Any:
    """Raise ``ValueError`` for any keys in *data* not declared on *cls*."""
    if isinstance(data, dict):
        valid_keys = _valid_field_keys(cls)
        for key in set(data.keys()) - valid_keys:
            hint = suggest_typo_fix(key, valid_keys)
            msg = f"Unknown {label} key '{key}'."
            if hint:
                msg += f" Did you mean '{hint}'?"
            raise ValueError(msg)
    return data


class BuildSpec(BaseModel):
    """Schema for an object build specification."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class_: str = Field(
        ...,
        alias="class",
        description="Fully-qualified Python class name.",
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Constructor keyword arguments.",
    )
    args: list[Any] = Field(
        default_factory=list,
        description="Constructor positional arguments.",
    )

    @model_validator(mode="before")
    @classmethod
    def _check_unknown_keys(cls, data: Any) -> Any:
        return _check_for_unknown_keys(cls, data, label="build spec")


class DataConfig(BaseModel):
    """Schema for data path configuration."""

    model_config = ConfigDict(extra="forbid")

    X_path: str | None = Field(None, description="Path to covariate data file.")
    y_path: str | None = Field(None, description="Path to target data file.")


class CalibrationStep(BaseModel):
    """Schema for a single calibration step."""

    model_config = ConfigDict(extra="forbid")

    method_name: str = Field(..., description="Calibration method name.")
    params: dict[str, Any] | None = Field(None, description="Method keyword arguments.")

    @model_validator(mode="before")
    @classmethod
    def _parse_single_method(cls, data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError(
                "Each calibration step must map exactly one method name "
                "to its parameters."
            )
        if "method_name" in data:
            return data
        if len(data) != 1:
            raise ValueError(
                "Each calibration step must map exactly one method name "
                "to its parameters."
            )
        name, params = next(iter(data.items()))
        return {"method_name": name, "params": params}


class MMMYamlConfig(BaseModel):
    """Schema for the top-level MMM YAML configuration."""

    model_config = ConfigDict(extra="forbid")

    model: BuildSpec = Field(..., description="MMM class and constructor arguments.")
    data: DataConfig | None = Field(None, description="Data file paths.")
    effects: list[BuildSpec] | None = Field(None, description="Additive effects.")
    original_scale_vars: list[str] | None = Field(
        None, description="Original-scale variables."
    )
    calibration: list[CalibrationStep] | None = Field(
        None, description="Calibration steps."
    )
    idata_path: str | None = Field(None, description="Path to InferenceData file.")

    @model_validator(mode="before")
    @classmethod
    def _check_unknown_keys(cls, data: Any) -> Any:
        return _check_for_unknown_keys(cls, data, label="config")

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> MMMYamlConfig:
        """Load and validate a YAML configuration file.

        Parameters
        ----------
        path : str or Path
            Path to the YAML configuration file.

        Returns
        -------
        MMMYamlConfig
            Validated configuration object.

        Raises
        ------
        pydantic.ValidationError
            If the YAML structure is invalid.
        FileNotFoundError
            If *path* does not exist.
        yaml.YAMLError
            If the file is not valid YAML.
        """
        raw = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(raw)
