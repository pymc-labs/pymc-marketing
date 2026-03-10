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
"""Scoring helpers for benchmark metrics and paired comparisons."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr


def _flatten_numeric_values(
    values: Mapping[str, Any], prefix: str = ""
) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in values.items():
        flat_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flat.update(_flatten_numeric_values(value, prefix=flat_key))
            continue
        if isinstance(value, (int, float, np.floating, np.integer)):
            flat[flat_key] = float(value)
    return flat


def _build_aligned_series(
    estimate: Mapping[str, Any],
    truth: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, float], list[str], list[str], list[str]]:
    est_flat = _flatten_numeric_values(estimate)
    true_flat = _flatten_numeric_values(truth)

    shared_keys = sorted(set(est_flat).intersection(true_flat))
    missing_estimate_keys = sorted(set(true_flat) - set(est_flat))
    missing_truth_keys = sorted(set(est_flat) - set(true_flat))

    # Fallback key alignment: map coarse estimate keys (e.g. x1/x2) to truth keys containing them.
    if not shared_keys and est_flat and true_flat:
        aligned_est: dict[str, float] = {}
        aligned_truth: dict[str, float] = {}
        for est_key, est_value in est_flat.items():
            matches = [
                truth_value
                for truth_key, truth_value in true_flat.items()
                if truth_key == est_key
                or truth_key.endswith(f".{est_key}")
                or f".{est_key}." in truth_key
            ]
            if matches:
                aligned_est[est_key] = est_value
                aligned_truth[est_key] = float(np.mean(matches))
        if aligned_est:
            shared_keys = sorted(aligned_est)
            missing_estimate_keys = []
            missing_truth_keys = []
            return (
                aligned_est,
                aligned_truth,
                shared_keys,
                missing_estimate_keys,
                missing_truth_keys,
            )

    aligned_est = {key: est_flat[key] for key in shared_keys}
    aligned_truth = {key: true_flat[key] for key in shared_keys}
    return (
        aligned_est,
        aligned_truth,
        shared_keys,
        missing_estimate_keys,
        missing_truth_keys,
    )


def convergence_from_sample_stats(sample_stats: xr.Dataset) -> dict[str, Any]:
    """Compute convergence diagnostics from sample stats."""
    if "diverging" not in sample_stats:
        return {
            "divergence_count": 0,
            "is_converged": True,
        }

    divergence_count = int(np.asarray(sample_stats["diverging"].values).sum())
    return {
        "divergence_count": divergence_count,
        "is_converged": divergence_count == 0,
    }


def compute_parameter_recovery_mae(
    estimate: Mapping[str, Any],
    truth: Mapping[str, Any],
) -> float:
    """Compute MAE between estimated and true parameters over shared keys."""
    details = compute_parameter_recovery_details(estimate=estimate, truth=truth)
    return float(details["mae"])


def compute_roas_recovery_mae(
    estimate: Mapping[str, float],
    truth: Mapping[str, float],
) -> float:
    """Compute ROAS MAE across shared channels."""
    details = compute_roas_recovery_details(estimate=estimate, truth=truth)
    return float(details["mae"])


def compute_parameter_recovery_details(
    estimate: Mapping[str, Any],
    truth: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute rich parameter recovery diagnostics."""
    aligned_est, aligned_truth, shared_keys, missing_estimate, missing_truth = (
        _build_aligned_series(estimate=estimate, truth=truth)
    )
    if not shared_keys:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "median_ae": float("nan"),
            "max_ae": float("nan"),
            "shared_key_count": 0,
            "missing_estimate_key_count": len(missing_estimate),
            "missing_truth_key_count": len(missing_truth),
            "error_by_key": {},
        }

    error_by_key = {
        key: abs(float(aligned_est[key]) - float(aligned_truth[key]))
        for key in shared_keys
    }
    errors = np.asarray(list(error_by_key.values()), dtype=np.float64)
    return {
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "median_ae": float(np.median(errors)),
        "max_ae": float(np.max(errors)),
        "shared_key_count": len(shared_keys),
        "missing_estimate_key_count": len(missing_estimate),
        "missing_truth_key_count": len(missing_truth),
        "error_by_key": error_by_key,
    }


def compute_roas_recovery_details(
    estimate: Mapping[str, float],
    truth: Mapping[str, float],
) -> dict[str, Any]:
    """Compute rich ROAS recovery diagnostics."""
    return compute_parameter_recovery_details(estimate=estimate, truth=truth)


def paired_delta(baseline_value: float, skilled_value: float) -> float:
    """Compute paired delta where negative means skilled is better for error metrics."""
    return float(skilled_value - baseline_value)


def aggregate_cv_crps(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Aggregate fold-level CRPS metrics into summary statistics."""
    crps_values = [float(item["crps"]) for item in fold_metrics if "crps" in item]
    if not crps_values:
        return {"crps_mean": float("nan"), "crps_std": float("nan"), "n_folds": 0.0}
    return {
        "crps_mean": float(np.mean(crps_values)),
        "crps_std": float(np.std(crps_values)),
        "n_folds": float(len(crps_values)),
    }


def aggregate_cv_train_test_crps(
    fold_metrics: list[dict[str, Any]],
) -> dict[str, float]:
    """Aggregate train/test CRPS across folds when available."""
    train_values = [
        float(item["crps_train"])
        for item in fold_metrics
        if item.get("crps_train") is not None
    ]
    test_values = [
        float(item["crps_test"])
        for item in fold_metrics
        if item.get("crps_test") is not None
    ]
    return {
        "train_crps_mean": float(np.mean(train_values))
        if train_values
        else float("nan"),
        "train_crps_std": float(np.std(train_values)) if train_values else float("nan"),
        "test_crps_mean": float(np.mean(test_values)) if test_values else float("nan"),
        "test_crps_std": float(np.std(test_values)) if test_values else float("nan"),
        "n_train_folds": float(len(train_values)),
        "n_test_folds": float(len(test_values)),
    }


def compute_generalization_gap(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute train-test CRPS generalization gap diagnostics."""
    gaps: list[float] = []
    for item in fold_metrics:
        train = item.get("crps_train")
        test = item.get("crps_test")
        if train is None or test is None:
            continue
        gaps.append(float(test) - float(train))
    if not gaps:
        return {
            "gap_mean": float("nan"),
            "gap_std": float("nan"),
            "gap_max": float("nan"),
            "gap_min": float("nan"),
            "n_folds": 0,
            "by_fold": [],
        }
    arr = np.asarray(gaps, dtype=np.float64)
    return {
        "gap_mean": float(np.mean(arr)),
        "gap_std": float(np.std(arr)),
        "gap_max": float(np.max(arr)),
        "gap_min": float(np.min(arr)),
        "n_folds": len(gaps),
        "by_fold": gaps,
    }


def summarize_cv_fold_diagnostics(
    cv_fold_diagnostics: list[dict[str, Any]],
) -> dict[str, float]:
    """Summarize fold-level convergence diagnostics."""
    if not cv_fold_diagnostics:
        return {
            "divergence_mean": float("nan"),
            "divergence_max": float("nan"),
            "rhat_max": float("nan"),
            "ess_bulk_min": float("nan"),
            "bfmi_min": float("nan"),
        }

    divergence = [
        float(item["divergence_count"])
        for item in cv_fold_diagnostics
        if item.get("divergence_count") is not None
    ]
    rhat = [
        float(item["rhat_max"])
        for item in cv_fold_diagnostics
        if item.get("rhat_max") is not None
    ]
    ess_bulk = [
        float(item["ess_bulk_min"])
        for item in cv_fold_diagnostics
        if item.get("ess_bulk_min") is not None
    ]
    bfmi = [
        float(item["bfmi_min"])
        for item in cv_fold_diagnostics
        if item.get("bfmi_min") is not None
    ]

    return {
        "divergence_mean": float(np.mean(divergence)) if divergence else float("nan"),
        "divergence_max": float(np.max(divergence)) if divergence else float("nan"),
        "rhat_max": float(np.max(rhat)) if rhat else float("nan"),
        "ess_bulk_min": float(np.min(ess_bulk)) if ess_bulk else float("nan"),
        "bfmi_min": float(np.min(bfmi)) if bfmi else float("nan"),
    }


def runtime_efficiency_metrics(runtime_sec: float, n_folds: int) -> dict[str, float]:
    """Compute runtime efficiency summaries for comparison dashboards."""
    folds = max(int(n_folds), 1)
    return {
        "runtime_sec": float(runtime_sec),
        "runtime_per_fold_sec": float(runtime_sec / folds),
        "folds_per_second": float(folds / runtime_sec)
        if runtime_sec > 0
        else float("nan"),
    }


def compute_cv_parameter_stability(
    cv_parameter_estimates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute per-parameter and aggregate stability statistics across CV folds."""
    if not cv_parameter_estimates:
        return {
            "parameter_count": 0,
            "param_std_mean": float("nan"),
            "param_iqr_mean": float("nan"),
            "param_cv_mean": float("nan"),
            "param_range_mean": float("nan"),
            "by_parameter": {},
        }

    parameter_values: dict[str, list[float]] = {}
    for fold in cv_parameter_estimates:
        values = _flatten_numeric_values(fold.get("parameter_estimates", {}))
        for key, value in values.items():
            parameter_values.setdefault(key, []).append(float(value))

    by_parameter: dict[str, dict[str, float]] = {}
    for key, values in parameter_values.items():
        if len(values) < 2:
            continue
        arr = np.asarray(values, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        q1 = float(np.quantile(arr, 0.25))
        q3 = float(np.quantile(arr, 0.75))
        iqr = float(q3 - q1)
        value_range = float(np.max(arr) - np.min(arr))
        coefvar = float(std / abs(mean)) if mean != 0 else float("nan")
        by_parameter[key] = {
            "mean": mean,
            "std": std,
            "iqr": iqr,
            "range": value_range,
            "coefvar": coefvar,
            "n_folds": float(len(arr)),
        }

    if not by_parameter:
        return {
            "parameter_count": 0,
            "param_std_mean": float("nan"),
            "param_iqr_mean": float("nan"),
            "param_cv_mean": float("nan"),
            "param_range_mean": float("nan"),
            "by_parameter": {},
        }

    std_values = [stats["std"] for stats in by_parameter.values()]
    iqr_values = [stats["iqr"] for stats in by_parameter.values()]
    cv_values = [
        stats["coefvar"]
        for stats in by_parameter.values()
        if np.isfinite(stats["coefvar"])
    ]
    range_values = [stats["range"] for stats in by_parameter.values()]
    return {
        "parameter_count": len(by_parameter),
        "param_std_mean": float(np.mean(std_values)),
        "param_iqr_mean": float(np.mean(iqr_values)),
        "param_cv_mean": float(np.mean(cv_values)) if cv_values else float("nan"),
        "param_range_mean": float(np.mean(range_values)),
        "by_parameter": by_parameter,
    }
