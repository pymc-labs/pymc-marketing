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
"""Ground-truth extraction helpers for benchmark tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from benchmark.schemas import BenchmarkTaskSpec


def _resolve_dataset_path(dataset_path: str) -> Path:
    path = Path(dataset_path)
    if path.exists():
        return path
    raise FileNotFoundError(f"Dataset path does not exist locally: {dataset_path}")


def _all_zero(values: dict[str, float]) -> bool:
    return bool(values) and all(float(value) == 0.0 for value in values.values())


def _logistic_saturation(values: np.ndarray, lam: float) -> np.ndarray:
    return (1.0 - np.exp(-lam * values)) / (1.0 + np.exp(-lam * values))


def _geometric_adstock(values: np.ndarray, alpha: float, l_max: int = 8) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    for t in range(len(values)):
        total = 0.0
        for lag in range(l_max):
            idx = t - lag
            if idx < 0:
                break
            total += (alpha**lag) * float(values[idx])
        result[t] = total
    return result


def _to_nested_dict(data_array: xr.DataArray) -> Any:
    if data_array.ndim == 0:
        return float(data_array.item())

    dim = data_array.dims[0]
    nested: dict[str, Any] = {}
    for coord in data_array.coords[dim].values:
        value = data_array.sel({dim: coord})
        nested[str(coord)] = _to_nested_dict(value)
    return nested


def load_true_parameters_from_netcdf(path: str | Path) -> dict[str, Any]:
    """Load true parameters saved by multidimensional data generation."""
    dataset = xr.open_dataset(path)
    payload: dict[str, Any] = {}
    for var_name, data_array in dataset.data_vars.items():
        payload[var_name] = _to_nested_dict(data_array)
    return payload


def _add_channel_level_parameter_proxies(parameters: dict[str, Any]) -> dict[str, Any]:
    """Add channel-level proxies to improve comparability with coarse estimates."""
    updated = dict(parameters)

    saturation_beta = parameters.get("saturation_beta")
    if isinstance(saturation_beta, dict):
        for channel, geo_values in saturation_beta.items():
            if isinstance(geo_values, dict):
                numeric_values = [
                    float(value)
                    for value in geo_values.values()
                    if isinstance(value, (int, float))
                ]
                if numeric_values:
                    updated[str(channel)] = float(np.mean(numeric_values))
                    updated[f"saturation_beta.{channel}"] = float(
                        np.mean(numeric_values)
                    )

    adstock_alpha = parameters.get("adstock_alpha")
    if isinstance(adstock_alpha, dict):
        for geo_values in adstock_alpha.values():
            if isinstance(geo_values, dict):
                for channel, value in geo_values.items():
                    if isinstance(value, (int, float)):
                        updated.setdefault(
                            f"adstock_alpha.{channel}",
                            float(value),
                        )
    return updated


def compute_true_roas_from_roas_dataset(
    dataset_path: str | Path,
    channel_columns: list[str],
) -> dict[str, float]:
    """Compute true ROAS from the ROAS confounding dataset equations."""
    df = pd.read_csv(_resolve_dataset_path(str(dataset_path)))
    roas: dict[str, float] = {}
    for idx, channel in enumerate(channel_columns, start=1):
        spend_sum = float(df[channel].sum())
        counterfactual_col = f"y{idx:02d}"
        if counterfactual_col in df.columns and spend_sum > 0:
            roas[channel] = float((df["y"] - df[counterfactual_col]).sum() / spend_sum)
            continue

        effect_col = f"{channel}_effect"
        if effect_col in df.columns and spend_sum > 0:
            roas[channel] = float(df[effect_col].sum() / spend_sum)
            continue

        roas[channel] = float("nan")
    return roas


def compute_true_parameters_from_roas_dataset(
    dataset_path: str | Path,
    channel_columns: list[str],
) -> dict[str, float]:
    """Estimate true effect coefficients directly from generated effect columns."""
    df = pd.read_csv(_resolve_dataset_path(str(dataset_path)))
    parameters: dict[str, float] = {}
    for channel in channel_columns:
        effect_col = f"{channel}_effect"
        transformed_col = f"{channel}_adstock_saturated"
        if effect_col not in df.columns or transformed_col not in df.columns:
            continue
        denom = df[transformed_col].replace(0.0, np.nan)
        ratio = (df[effect_col] / denom).replace([np.inf, -np.inf], np.nan).dropna()
        if ratio.empty:
            continue
        parameters[f"beta_{channel}_adstock_saturated"] = float(ratio.median())
    return parameters


def _extract_param_scalar(
    data_array: xr.DataArray,
    channel: str,
    geo: str,
) -> float:
    selectors: dict[str, str] = {}
    if "channel" in data_array.dims:
        selectors["channel"] = channel
    if "geo" in data_array.dims:
        selectors["geo"] = geo
    return float(data_array.sel(selectors).item())


def compute_true_roas_from_multidimensional_data(
    dataset_path: str | Path,
    parameters_netcdf_path: str | Path,
    channel_columns: list[str],
    date_column: str = "date",
    geo_column: str = "geo",
    l_max: int = 8,
) -> dict[str, float]:
    """Compute channel ROAS from multidimensional generator parameters and spends."""
    df = pd.read_csv(_resolve_dataset_path(str(dataset_path)))
    true_params = xr.open_dataset(parameters_netcdf_path)
    roas: dict[str, float] = {}
    for channel in channel_columns:
        total_contribution = 0.0
        total_spend = 0.0
        for geo, sub_df in df.groupby(geo_column):
            sub_df = sub_df.sort_values(date_column)
            spend = sub_df[channel].to_numpy(dtype=np.float64)
            alpha = _extract_param_scalar(
                true_params["adstock_alpha"], channel=channel, geo=str(geo)
            )
            lam = _extract_param_scalar(
                true_params["saturation_lam"], channel=channel, geo=str(geo)
            )
            beta = _extract_param_scalar(
                true_params["saturation_beta"], channel=channel, geo=str(geo)
            )
            adstocked = _geometric_adstock(spend, alpha=alpha, l_max=l_max)
            contribution = beta * _logistic_saturation(adstocked, lam=lam)
            total_contribution += float(np.sum(contribution))
            total_spend += float(np.sum(spend))
        roas[channel] = (
            float(total_contribution / total_spend) if total_spend > 0 else float("nan")
        )
    return roas


def resolve_task_ground_truth(
    task: BenchmarkTaskSpec,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Resolve task ground-truth parameters/ROAS from config and source artifacts."""
    if task.ground_truth is None:
        return {}, {}

    parameters_truth = dict(task.ground_truth.parameters)
    roas_truth = dict(task.ground_truth.roas)

    if task.task_type == "mmm_multidimensional":
        source_file = parameters_truth.get("source_file")
        if isinstance(source_file, str):
            parameters_truth = load_true_parameters_from_netcdf(source_file)
            parameters_truth = _add_channel_level_parameter_proxies(parameters_truth)
            if not roas_truth or _all_zero(roas_truth):
                roas_truth = compute_true_roas_from_multidimensional_data(
                    dataset_path=task.dataset_path,
                    parameters_netcdf_path=source_file,
                    channel_columns=task.channel_columns,
                    date_column=task.date_column,
                )

    if task.task_type == "mmm_roas_confounding":
        if not parameters_truth or _all_zero(
            {
                k: float(v)
                for k, v in parameters_truth.items()
                if isinstance(v, (int, float))
            }
        ):
            parameters_truth = compute_true_parameters_from_roas_dataset(
                dataset_path=task.dataset_path,
                channel_columns=task.channel_columns,
            )
        if not roas_truth or _all_zero(roas_truth):
            roas_truth = compute_true_roas_from_roas_dataset(
                dataset_path=task.dataset_path,
                channel_columns=task.channel_columns,
            )

    return parameters_truth, roas_truth
