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
"""Builder for MMM projects."""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml  # type: ignore

from pymc_marketing.mmm.builders.factories import build, resolve
from pymc_marketing.mmm.multidimensional import MMM
from pymc_marketing.utils import from_netcdf


def _load_df(path: str | Path) -> pd.DataFrame:
    """
    Read a DataFrame from *path* based on extension.

    Currently supports: .parquet, .csv, .txt
    """
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unrecognised tabular format: {path}")


def _apply_and_validate_calibration_steps(
    model: MMM, cfg: Mapping[str, Any], base_dir: Path
) -> None:
    calibration_specs = cfg.get("calibration", [])
    if not calibration_specs:
        return

    if not isinstance(calibration_specs, list):
        raise TypeError("`calibration` section must be a list of steps.")

    for step in calibration_specs:
        if not isinstance(step, Mapping):
            raise TypeError(
                "Each calibration step must be a mapping of method to parameters."
            )
        if len(step) != 1:
            raise ValueError(
                "Calibration steps must map a single method to its parameters."
            )

        method_name, raw_params = next(iter(step.items()))

        if not hasattr(model, method_name):
            raise AttributeError(f"MMM has no calibration method '{method_name}'.")

        method = getattr(model, method_name)
        if not callable(method):
            raise TypeError(f"Attribute '{method_name}' is not callable on MMM.")

        if raw_params is not None and not isinstance(raw_params, Mapping):
            raise TypeError(
                f"Calibration parameters for '{method_name}' must be a mapping, got {type(raw_params).__name__}."
            )

        if (
            method_name == "add_lift_test_measurements"
            and raw_params
            and "dist" in raw_params
        ):
            raise ValueError(
                "`dist` parameter for 'add_lift_test_measurements' is not supported via YAML configuration yet."
            )

        resolved_kwargs = (
            {key: resolve(value) for key, value in raw_params.items()}
            if raw_params is not None
            else {}
        )

        try:
            method(**resolved_kwargs)
        except Exception as err:  # pragma: no cover - re-raise with context
            raise RuntimeError(
                f"Failed to apply calibration step '{method_name}' from YAML configuration: \n {err}"
            ) from err


def build_mmm_from_yaml(
    config_path: str | Path,
    *,
    X: pd.DataFrame | None = None,
    y: pd.DataFrame | pd.Series | None = None,
    model_kwargs: dict | None = None,
) -> MMM:
    """
    Build an MMM model from *config_path*.

    The configuration keys:

    - `model` (required): MMM initialization parameters
    - `effects` (optional): list of additive effects in the model
    - `data` (optional): paths to X and y data
    - `original_scale_vars` (optional): list of original scale variables
    - `idata_path` (optional): path to inference data

    Parameters
    ----------
    config_path : str | Path
        YAML file with model configuration.
    X : pandas.DataFrame, optional
        Pre-loaded covariate matrix.  If omitted, the loader tries to read it
        from a path in the YAML under `data.X_path`.
    y : pandas.DataFrame | pandas.Series, optional
        Pre-loaded target vector.  If omitted, the loader tries to read it
        from a path in the YAML under `data.y_path`.
    model_kwargs : dict, optional
        Additional keyword arguments for the model.
        They override any defaults specified in the YAML config.

    Returns
    -------
    model : MMM
    """
    config_path = Path(config_path)
    cfg: Mapping[str, Any] = yaml.safe_load(config_path.read_text())

    # 1 ─────────────────────────────────── shell (no effects yet)
    # Merge model_kwargs into cfg["model"]["kwargs"], with model_kwargs taking precedence
    model_config = {**cfg["model"].get("kwargs", {}), **(model_kwargs or {})}
    cfg["model"]["kwargs"] = model_config

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model = build(cfg["model"])

    # 2 ──────────────────────────────── resolve covariates / target
    data_cfg: Mapping[str, Any] = cfg.get("data", {})
    if X is None:
        if "X_path" not in data_cfg:
            raise ValueError("X not provided and no `data.X_path` found in YAML.")
        X = _load_df(data_cfg["X_path"])
    if y is None:
        if "y_path" not in data_cfg:
            raise ValueError("y not provided and no `data.y_path` found in YAML.")
        y = _load_df(data_cfg["y_path"])

    # Convert date column after loading data
    date_column = model_config.get("date_column")
    if date_column:
        date_col_in_X = date_column in X.columns

        if date_column in X.columns:
            X[date_column] = pd.to_datetime(X[date_column])

        if not date_col_in_X:
            raise ValueError(
                f"Date column '{date_column}' specified in config not found in either X or y data."
            )

    # 3 ───────────────────────────────────── effects (preserve order)
    # Build and append each effect
    for eff_spec in cfg.get("effects", []):
        effect = build(eff_spec)
        model.mu_effects.append(effect)

    # 4 ───────────────────────────────────────────── build PyMC graph
    model.build_model(X, y)  # this **must** precede any idata loading

    # 4b ──────────────────────────────────────────── apply calibration steps (if any)
    _apply_and_validate_calibration_steps(model, cfg, config_path.parent)

    # 5 ───────────────────────── add original scale contribution variables
    original_scale_vars = cfg.get("original_scale_vars", [])
    if original_scale_vars:
        model.add_original_scale_contribution_variable(var=original_scale_vars)

    # 6 ──────────────────────────────────────────── attach inference data
    if (idata_fp := cfg.get("idata_path")) is not None:
        idata_path = Path(idata_fp)
        if os.path.exists(idata_path):
            model.idata = from_netcdf(idata_path)

    return model
