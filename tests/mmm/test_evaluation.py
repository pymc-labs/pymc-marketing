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

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_marketing.mmm.evaluation import (
    calculate_metric_distributions,
    compute_summary_metrics,
    summarize_metric_distributions,
)


@pytest.fixture(scope="function")
def manage_random_state():
    """Fixture to manage random state before and after each test."""
    # Setup: save current state and set seed
    original_state = np.random.get_state()
    np.random.seed(42)

    yield

    # Teardown: restore original state
    np.random.set_state(original_state)


@pytest.fixture
def sample_data(manage_random_state) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Create 100 samples of predictions with some noise (seed has been set)
    y_pred = np.array([y_true + np.random.normal(0, 0.1, 5) for _ in range(100)]).T
    return y_true, y_pred


@pytest.mark.parametrize("y_true_cls", [np.array, pd.Series])
@pytest.mark.parametrize("y_pred_cls", [np.array, xr.DataArray])
def test_calculate_metric_distributions_all_metrics(
    sample_data,
    y_true_cls,
    y_pred_cls,
) -> None:
    y_true, y_pred = sample_data
    y_true = y_true_cls(y_true)
    y_pred = y_pred_cls(y_pred)

    metrics = ["r_squared", "rmse", "nrmse", "mae", "nmae", "mape"]

    results = calculate_metric_distributions(y_true, y_pred, metrics)

    # Basic structure checks
    assert isinstance(results, dict)
    assert all(metric in results for metric in metrics)
    assert all(isinstance(results[metric], np.ndarray) for metric in metrics)
    assert all(len(results[metric]) == y_pred.shape[1] for metric in metrics)

    # Value range checks
    assert all(0 <= results["r_squared"]) and all(results["r_squared"] <= 1)
    assert all(results["rmse"] >= 0)
    assert all(results["nrmse"] >= 0)
    assert all(results["mae"] >= 0)
    assert all(results["nmae"] >= 0)
    assert all(results["mape"] >= 0)


def test_calculate_metric_distributions_default_metrics(sample_data) -> None:
    """Test that default metrics are used when metrics_to_calculate is None."""
    y_true, y_pred = sample_data

    results = calculate_metric_distributions(y_true, y_pred)

    # Should include all default metrics
    expected_metrics = ["r_squared", "rmse", "nrmse", "mae", "nmae", "mape"]
    assert all(metric in results for metric in expected_metrics)


def test_calculate_metric_distributions_invalid_metrics() -> None:
    """Test that invalid metrics are caught and reported."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([[1.1, 2.1], [1.0, 2.0]]).T

    invalid_metrics = ["invalid_metric_1", "smape", "crps"]

    with pytest.raises(ValueError):
        calculate_metric_distributions(y_true, y_pred, invalid_metrics)


def test_summarize_metric_distributions(sample_data) -> None:
    y_true, y_pred = sample_data
    metric_distributions = calculate_metric_distributions(y_true, y_pred)

    # Test with different HDI probability
    summaries_89 = summarize_metric_distributions(metric_distributions, hdi_prob=0.89)
    summaries_95 = summarize_metric_distributions(metric_distributions, hdi_prob=0.95)

    expected_stats = [
        "mean",
        "median",
        "std",
        "min",
        "max",
        "89%_hdi_lower",
        "89%_hdi_upper",
    ]

    # Basic structure checks
    assert isinstance(summaries_89, dict)
    assert all(metric in summaries_89 for metric in metric_distributions.keys())
    assert all(
        all(stat in summaries_89[metric] for stat in expected_stats)
        for metric in summaries_89
    )

    # Check that HDI bounds are different for different probabilities
    for metric in metric_distributions:
        assert (
            summaries_89[metric]["89%_hdi_lower"]
            >= summaries_95[metric]["95%_hdi_lower"]
        )
        assert (
            summaries_89[metric]["89%_hdi_upper"]
            <= summaries_95[metric]["95%_hdi_upper"]
        )


def test_compute_summary_metrics(sample_data) -> None:
    y_true, y_pred = sample_data
    metrics = ["r_squared", "rmse", "mae"]

    results = compute_summary_metrics(
        y_true, y_pred, metrics_to_calculate=metrics, hdi_prob=0.89
    )

    expected_stats = [
        "mean",
        "median",
        "std",
        "min",
        "max",
        "89%_hdi_lower",
        "89%_hdi_upper",
    ]
    # Structure checks
    assert isinstance(results, dict)
    assert all(metric in results for metric in metrics)
    assert all(
        all(stat in results[metric] for stat in expected_stats) for metric in results
    )
    # Value consistency checks
    for metric in metrics:
        assert (
            results[metric]["min"] <= results[metric]["mean"] <= results[metric]["max"]
        )
        assert (
            results[metric]["89%_hdi_lower"]
            <= results[metric]["median"]
            <= results[metric]["89%_hdi_upper"]
        )
        assert results[metric]["std"] >= 0


@pytest.mark.parametrize(
    argnames="y_true, y_pred",
    argvalues=[
        (
            np.array([10.0, 25.0, 50.0, 100.0, 150.0, 200.0, 75.0, 15.0]),
            np.array(
                [
                    [
                        12.0,
                        28.0,
                        45.0,
                        110.0,
                        140.0,
                        190.0,
                        80.0,
                        18.0,
                    ],  # Slight overestimate
                    [
                        8.0,
                        22.0,
                        55.0,
                        90.0,
                        160.0,
                        210.0,
                        70.0,
                        12.0,
                    ],  # Mixed under/over
                    [
                        11.0,
                        26.0,
                        48.0,
                        105.0,
                        145.0,
                        195.0,
                        77.0,
                        16.0,
                    ],  # Close to true
                    [5.0, 20.0, 40.0, 85.0, 130.0, 180.0, 65.0, 10.0],  # Underestimate
                ]
            ).T,
        ),
    ],
)
def test_metric_consistency_across_functions(y_true, y_pred) -> None:
    """Test that metrics are consistent between direct calculation and distribution summary."""

    # Calculate metrics directly
    distributions = calculate_metric_distributions(y_true, y_pred)
    summaries = summarize_metric_distributions(distributions)

    # Calculate via compute_summary_metrics
    direct_summaries = compute_summary_metrics(y_true, y_pred)

    # Compare results
    for metric in distributions:
        assert summaries[metric]["mean"] == pytest.approx(
            direct_summaries[metric]["mean"], rel=1e-10
        )
