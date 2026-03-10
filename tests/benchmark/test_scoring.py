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
import numpy as np
import pytest
import xarray as xr

from benchmark.scoring import (
    aggregate_cv_crps,
    compute_cv_parameter_stability,
    compute_parameter_recovery_details,
    compute_parameter_recovery_mae,
    compute_roas_recovery_details,
    compute_roas_recovery_mae,
    convergence_from_sample_stats,
    paired_delta,
)


def test_convergence_from_sample_stats_counts_divergences() -> None:
    sample_stats = xr.Dataset(
        data_vars={
            "diverging": (("chain", "draw"), np.array([[0, 1], [0, 0]], dtype=np.int64))
        }
    )

    convergence = convergence_from_sample_stats(sample_stats)
    assert convergence["divergence_count"] == 1
    assert convergence["is_converged"] is False


def test_parameter_recovery_mae() -> None:
    estimate = {"x1": 0.15, "x2": 0.4}
    truth = {"x1": 0.2, "x2": 0.3}
    mae = compute_parameter_recovery_mae(estimate, truth)
    assert mae == pytest.approx((0.05 + 0.1) / 2)


def test_roas_recovery_mae() -> None:
    estimate = {"x1": 1.3, "x2": 1.0}
    truth = {"x1": 1.1, "x2": 0.9}
    mae = compute_roas_recovery_mae(estimate, truth)
    assert mae == pytest.approx((0.2 + 0.1) / 2)


def test_paired_delta_computes_skilled_minus_baseline() -> None:
    delta = paired_delta(baseline_value=0.8, skilled_value=0.6)
    assert delta == pytest.approx(-0.2)


def test_aggregate_cv_crps() -> None:
    summary = aggregate_cv_crps(
        [
            {"fold_idx": 0, "crps": 0.10},
            {"fold_idx": 1, "crps": 0.20},
            {"fold_idx": 2, "crps": 0.15},
        ]
    )
    assert summary["crps_mean"] == pytest.approx(0.15)
    assert summary["n_folds"] == pytest.approx(3.0)


def test_parameter_recovery_details() -> None:
    details = compute_parameter_recovery_details(
        estimate={"x1": 0.1, "x2": 0.2},
        truth={"x1": 0.2, "x2": 0.1},
    )
    assert details["shared_key_count"] == 2
    assert details["mae"] == pytest.approx(0.1)
    assert "x1" in details["error_by_key"]


def test_roas_recovery_details() -> None:
    details = compute_roas_recovery_details(
        estimate={"x1": 1.2, "x2": 0.8},
        truth={"x1": 1.0, "x2": 1.0},
    )
    assert details["rmse"] > 0
    assert details["max_ae"] == pytest.approx(0.2)


def test_cv_parameter_stability() -> None:
    stability = compute_cv_parameter_stability(
        [
            {"fold_idx": 0, "parameter_estimates": {"x1": 0.20, "x2": 0.30}},
            {"fold_idx": 1, "parameter_estimates": {"x1": 0.22, "x2": 0.29}},
            {"fold_idx": 2, "parameter_estimates": {"x1": 0.21, "x2": 0.31}},
        ]
    )
    assert stability["parameter_count"] == 2
    assert stability["param_std_mean"] > 0
