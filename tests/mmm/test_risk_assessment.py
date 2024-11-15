#   Copyright 2024 The PyMC Labs Developers
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
import pymc as pm
import pytensor.tensor as pt
import pytest
from pytensor import function

from pymc_marketing.mmm.risk_assessment import (
    _compute_quantile,
    _covariance_matrix,
    mean_tightness_score,
    tail_distance,
)

rng: np.random.Generator = np.random.default_rng(seed=42)


@pytest.mark.parametrize(
    "mean1, std1, mean2, std2, expected_order",
    [
        (
            100,
            30,
            100,
            50,
            "greater",
        ),  # Expect greater tail distance for higher std deviation
        (
            100,
            30,
            100,
            10,
            "smaller",
        ),  # Expect smaller tail distance for lower std deviation
    ],
)
def test_tail_distance(mean1, std1, mean2, std2, expected_order):
    # Generate samples for both distributions
    samples1 = pm.Normal.dist(mu=mean1, sigma=std1, size=100).eval()
    samples2 = pm.Normal.dist(mu=mean2, sigma=std2, size=100).eval()

    # Calculate tail distances
    tail_distance_func = tail_distance(confidence_level=0.75)
    tail_distance1 = tail_distance_func(samples1, None).eval()
    tail_distance2 = tail_distance_func(samples2, None).eval()

    # Check that the tail distance is greater for the higher std deviation
    if expected_order == "greater":
        assert (
            tail_distance2 > tail_distance1
        ), f"Expected tail distance to be greater for std={std2}, but got {tail_distance2} <= {tail_distance1}"
    elif expected_order == "smaller":
        assert (
            tail_distance1 > tail_distance2
        ), f"Expected tail distance to be greater for std={std1}, but got {tail_distance1} <= {tail_distance2}"


@pytest.mark.parametrize(
    "mean1, std1, mean2, std2, alpha, expected_relation",
    [
        (
            100,
            30,
            120,
            60,
            0.9,
            "lower_std",
        ),  # With high alpha, lower std should dominate
        (
            100,
            30,
            120,
            60,
            0.1,
            "higher_mean",
        ),  # With low alpha, higher mean should dominate
    ],
)
def test_mean_tightness_score(mean1, std1, mean2, std2, alpha, expected_relation):
    # Generate samples for both distributions
    samples1 = pm.Normal.dist(mu=mean1, sigma=std1, size=100).eval()
    samples2 = pm.Normal.dist(mu=mean2, sigma=std2, size=100).eval()

    # Calculate mean tightness scores
    mean_tightness_score_func = mean_tightness_score(alpha=alpha, confidence_level=0.75)
    score1 = mean_tightness_score_func(samples1, None).eval()
    score2 = mean_tightness_score_func(samples2, None).eval()

    # Assertions based on observed behavior: higher mean should dominate in both cases
    if expected_relation == "higher_mean":
        assert (
            score2 > score1
        ), f"Expected score for mean={mean2} to be higher, but got {score2} <= {score1}"
    elif expected_relation == "lower_std":
        assert (
            score1 > score2
        ), f"Expected score for std={std1} to be lower, but got {score1} <= {score2}"


@pytest.mark.parametrize(
    "data, quantile",
    [
        ([1, 2, 3, 4, 5], 0.25),
        ([1, 2, 3, 4, 5], 0.5),
        ([1, 2, 3, 4, 5], 0.75),
        ([10, 20, 30, 40, 50], 0.1),
        ([10, 20, 30, 40, 50], 0.9),
        ([-5, -1, 0, 1, 5], 0.5),
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 0.33),
        ([100], 0.5),  # Single-element edge case
        ([1, 2], 0.5),  # Small array edge case
    ],
)
def test_compute_quantile_matches_numpy(data, quantile):
    # Convert data to NumPy array
    np_data = np.array(data)

    # Define symbolic variable for input
    pt_data = pt.vector("pt_data")  # Symbolic variable for 1D input data

    # Compile the PyTensor quantile function
    pt_quantile_func = function([pt_data], _compute_quantile(pt_data, quantile))

    # Compute results
    pytensor_result = pt_quantile_func(np_data)  # Pass NumPy array here
    numpy_result = np.quantile(np_data, quantile)

    # Assert the results are close
    np.testing.assert_allclose(
        pytensor_result,
        numpy_result,
        rtol=1e-3,
        atol=1e-8,
        err_msg=f"Mismatch for data={data} and quantile={quantile}",
    )


@pytest.mark.parametrize(
    "data",
    [
        np.array([[1, 2], [3, 4], [5, 6]]),  # Small test case
        np.random.rand(100, 10),  # Random large dataset
        np.array([[1, 1], [1, 1], [1, 1]]),  # Identical columns (zero variance)
        np.array([[1], [2], [3]]),  # Single-column case
    ],
)
def test_covariance_matrix_matches_numpy(data):
    # Define symbolic variable for input
    pt_data = pt.matrix("pt_data")  # Symbolic variable for 2D input data

    # Compile the PyTensor covariance matrix function
    pt_cov_func = function([pt_data], _covariance_matrix(pt_data))

    # Compute results
    pytensor_result = pt_cov_func(data)  # Pass NumPy array directly
    numpy_result = np.cov(data, rowvar=False)

    # Assert the results are close
    np.testing.assert_allclose(
        pytensor_result,
        numpy_result,
        rtol=1e-5,
        atol=1e-8,
        err_msg=f"Mismatch for input data:\n{data}",
    )
