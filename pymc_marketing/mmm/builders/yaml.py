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
"""Builder for MMM projects."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import pandas as pd

from pymc_marketing.mmm.builders.factories import build, resolve
from pymc_marketing.mmm.builders.schema import CalibrationStep, MMMYamlConfig
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
    model: MMM,
    steps: list[CalibrationStep] | None,
) -> None:
    for step in steps or []:
        if not hasattr(model, step.method_name):
            raise AttributeError(f"MMM has no calibration method '{step.method_name}'.")

        method = getattr(model, step.method_name)
        if not callable(method):
            raise TypeError(f"Attribute '{step.method_name}' is not callable on MMM.")

        if (
            step.method_name == "add_lift_test_measurements"
            and step.params
            and "dist" in step.params
        ):
            raise ValueError(
                "`dist` parameter for 'add_lift_test_measurements' is not "
                "supported via YAML configuration yet."
            )

        resolved_kwargs = (
            {key: resolve(value) for key, value in step.params.items()}
            if step.params is not None
            else {}
        )

        try:
            method(**resolved_kwargs)
        except Exception as err:  # pragma: no cover - re-raise with context
            raise RuntimeError(
                f"Failed to apply calibration step '{step.method_name}' "
                f"from YAML configuration: \n {err}"
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
    cfg = MMMYamlConfig.from_yaml_file(config_path)

    # 1 -- build MMM shell (no effects yet)
    model_spec = cfg.model.model_dump(by_alias=True)
    model_spec["kwargs"] = {**model_spec.get("kwargs", {}), **(model_kwargs or {})}

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        model = build(model_spec)

    # 2 -- resolve covariates / target
    data_cfg = cfg.data
    if X is None:
        if data_cfg is None or data_cfg.X_path is None:
            raise ValueError("X not provided and no `data.X_path` found in YAML.")
        X = _load_df(data_cfg.X_path)
    if y is None:
        if data_cfg is None or data_cfg.y_path is None:
            raise ValueError("y not provided and no `data.y_path` found in YAML.")
        y = _load_df(data_cfg.y_path)

    date_column = model_spec["kwargs"].get("date_column")
    if date_column:
        date_col_in_X = date_column in X.columns

        if date_column in X.columns:
            X.loc[:, date_column] = pd.to_datetime(X[date_column])

        if not date_col_in_X:
            raise ValueError(
                f"Date column '{date_column}' specified in config not found "
                f"in either X or y data."
            )

    # 3 -- effects (preserve order)
    for eff_spec in cfg.effects or []:
        effect = build(eff_spec.model_dump(by_alias=True))
        model.add_mu_effect(effect)

    # 4 -- build PyMC graph (must precede idata loading)
    model.build_model(X, y)

    # 5 -- add original scale contribution variables
    original_scale_vars = cfg.original_scale_vars or []
    if original_scale_vars:
        model.add_original_scale_contribution_variable(var=original_scale_vars)

    # 6 -- apply calibration steps (if any)
    _apply_and_validate_calibration_steps(model, cfg.calibration)

    # 7 -- attach inference data
    if cfg.idata_path is not None:
        idata_path = Path(cfg.idata_path)
        if os.path.exists(idata_path):
            model.idata = from_netcdf(idata_path)

    return model
