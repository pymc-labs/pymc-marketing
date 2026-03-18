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
"""Pydantic schemas for validating MMM YAML configuration files.

Mirrors the ``validate_or_raise`` / factory-classmethod patterns from
:mod:`pymc_marketing.data.idata.schema`.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any

import yaml  # type: ignore
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


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


_BUILD_SPEC_KEYS = {"class", "class_", "kwargs", "args"}


class BuildSpec(BaseModel):
    """Schema for an object build specification.

    Parameters
    ----------
    class_ : str
        Fully-qualified Python class name (aliased from ``class`` in YAML).
    kwargs : dict, default {}
        Keyword arguments for the class constructor.  Intentionally
        ``dict[str, Any]`` because values are heterogeneous (plain values,
        nested build specs, priors, etc.).
    args : list, default []
        Positional arguments for the class constructor.

    Examples
    --------
    >>> spec = BuildSpec.model_validate(
    ...     {"class": "pymc_marketing.mmm.GeometricAdstock", "kwargs": {"l_max": 12}}
    ... )
    >>> spec.class_
    'pymc_marketing.mmm.GeometricAdstock'
    """

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
        if isinstance(data, dict):
            for key in set(data.keys()) - _BUILD_SPEC_KEYS:
                hint = suggest_typo_fix(key, _BUILD_SPEC_KEYS)
                msg = f"Unknown key '{key}' in build spec."
                if hint:
                    msg += f" Did you mean '{hint}'?"
                raise ValueError(msg)
        return data


class DataConfig(BaseModel):
    """Schema for data path configuration.

    Parameters
    ----------
    X_path : str or None
        Path to the covariate data file.
    y_path : str or None
        Path to the target data file.
    """

    model_config = ConfigDict(extra="forbid")

    X_path: str | None = Field(None, description="Path to covariate data file.")
    y_path: str | None = Field(None, description="Path to target data file.")


class CalibrationStep(BaseModel):
    """Schema for a single calibration step.

    In YAML this is written as a single-key mapping, e.g.::

        - add_lift_test_measurements:
            df_lift_test: ...

    The ``model_validator`` restructures this into ``method_name`` /
    ``params`` fields.

    Parameters
    ----------
    method_name : str
        Name of the calibration method to call on the MMM.
    params : dict or None
        Keyword arguments to pass to the method.
    """

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


_TOP_LEVEL_KEYS = {
    "model",
    "data",
    "effects",
    "original_scale_vars",
    "calibration",
    "idata_path",
}


class MMMYamlConfig(BaseModel):
    """Schema for the top-level MMM YAML configuration.

    Mirrors the ``validate_or_raise`` pattern from
    :class:`pymc_marketing.data.idata.schema.MMMIdataSchema`.

    Parameters
    ----------
    model : BuildSpec
        MMM class and constructor arguments (required).
    data : DataConfig or None
        Paths to covariate and target data files.
    effects : list of BuildSpec or None
        Additive effects to append to the model.
    original_scale_vars : list of str or None
        Variables to add at original scale.
    calibration : list of CalibrationStep or None
        Calibration steps to apply after model build.
    idata_path : str or None
        Path to pre-existing InferenceData (NetCDF).

    Examples
    --------
    >>> cfg = MMMYamlConfig.from_yaml_file("config.yml")  # doctest: +SKIP
    >>> cfg.model.class_  # doctest: +SKIP
    'pymc_marketing.mmm.multidimensional.MMM'
    """

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
        if isinstance(data, dict):
            for key in set(data.keys()) - _TOP_LEVEL_KEYS:
                hint = suggest_typo_fix(key, _TOP_LEVEL_KEYS)
                msg = f"Unknown config key '{key}'."
                if hint:
                    msg += f" Did you mean '{hint}'?"
                raise ValueError(msg)
        return data

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> MMMYamlConfig:
        """Load and validate a YAML configuration file.

        Analogous to
        :meth:`pymc_marketing.data.idata.schema.MMMIdataSchema.from_model_config`.

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

    @classmethod
    def validate_or_raise(cls, raw: dict[str, Any]) -> MMMYamlConfig:
        """Validate a raw config dict, re-raising as :class:`ValueError`.

        Mirrors
        :meth:`pymc_marketing.data.idata.schema.MMMIdataSchema.validate_or_raise`.

        Parameters
        ----------
        raw : dict
            Raw parsed YAML dictionary.

        Returns
        -------
        MMMYamlConfig
            Validated configuration object.

        Raises
        ------
        ValueError
            If validation fails, with detailed error messages including
            field locations.
        """
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            parts: list[str] = []
            for e in exc.errors():
                loc = " -> ".join(str(x) for x in e["loc"])
                line = f"  - {e['msg']}"
                if loc:
                    line += f" (at {loc})"
                parts.append(line)
            error_msg = "YAML configuration validation failed:\n" + "\n".join(parts)
            raise ValueError(error_msg) from exc
