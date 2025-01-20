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
import pymc as pm
import pytensor.tensor as pt
import pytest
from pytensor import function

from pymc_marketing.mmm.utility import (
    _calculate_roas_distribution_for_allocation,
    _compute_quantile,
    _covariance_matrix,
    adjusted_value_at_risk_score,
    average_response,
    conditional_value_at_risk,
    mean_tightness_score,
    portfolio_entropy,
    raroc,
    sharpe_ratio,
    tail_distance,
    value_at_risk,
)

rng: np.random.Generator = np.random.default_rng(seed=42)

EXPECTED_RESULTS = {
    "avg_response": 5.5,
    "tail_dist": 4.5,
    "mean_tight_score": 3.25,
    "var_95": 1.45,
    "cvar_95": 1.0,
    "sharpe": 1.81327,
    "raroc_value": 0.00099891,
    "adjusted_var": 2.26,
    "entropy": 2.15181,
}


@pytest.fixture
def test_data():
    """
    Fixture to generate consistent test data for all tests.
    """
    samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    budgets = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    return pt.as_tensor_variable(samples), pt.as_tensor_variable(budgets)


def test_mean_tightness_score(test_data):
    samples, budgets = test_data
    result = mean_tightness_score(0.5, 0.75)(samples, budgets).eval()
    np.testing.assert_almost_equal(
        result,
        EXPECTED_RESULTS["mean_tight_score"],
        decimal=3,
        err_msg=f"Mean Tightness Score mismatch: {result} != {EXPECTED_RESULTS['mean_tight_score']}",
    )


def test_value_at_risk(test_data):
    samples, budgets = test_data
    result = value_at_risk(0.95)(samples, budgets).eval()
    np.testing.assert_almost_equal(
        result,
        EXPECTED_RESULTS["var_95"],
        decimal=3,
        err_msg=f"Value at Risk mismatch: {result} != {EXPECTED_RESULTS['var_95']}",
    )


def test_conditional_value_at_risk(test_data):
    samples, budgets = test_data
    result = conditional_value_at_risk(0.95)(samples, budgets).eval()
    np.testing.assert_almost_equal(
        result,
        EXPECTED_RESULTS["cvar_95"],
        decimal=3,
        err_msg=f"Conditional Value at Risk mismatch: {result} != {EXPECTED_RESULTS['cvar_95']}",
    )


def test_sharpe_ratio(test_data):
    samples, budgets = test_data
    result = sharpe_ratio(0.01)(samples, budgets).eval()
    np.testing.assert_almost_equal(
        result,
        EXPECTED_RESULTS["sharpe"],
        decimal=3,
        err_msg=f"Sharpe Ratio mismatch: {result} != {EXPECTED_RESULTS['sharpe']}",
    )


def test_raroc(test_data):
    samples, budgets = test_data
    result = raroc(0.01)(samples, budgets).eval()
    np.testing.assert_almost_equal(
        result,
        EXPECTED_RESULTS["raroc_value"],
        decimal=3,
        err_msg=f"RAROC mismatch: {result} != {EXPECTED_RESULTS['raroc_value']}",
    )


def test_adjusted_value_at_risk_score(test_data):
    samples, budgets = test_data
    result = adjusted_value_at_risk_score(0.95, 0.8)(samples, budgets).eval()
    np.testing.assert_almost_equal(
        result,
        EXPECTED_RESULTS["adjusted_var"],
        decimal=3,
        err_msg=f"Adjusted Value at Risk mismatch: {result} != {EXPECTED_RESULTS['adjusted_var']}",
    )


def test_portfolio_entropy(test_data):
    samples, budgets = test_data
    result = portfolio_entropy(samples, budgets).eval()
    np.testing.assert_almost_equal(
        result,
        EXPECTED_RESULTS["entropy"],
        decimal=3,
        err_msg=f"Portfolio Entropy mismatch: {result} != {EXPECTED_RESULTS['entropy']}",
    )


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
    samples1 = pt.as_tensor(
        pm.draw(pm.Normal.dist(mu=mean1, sigma=std1, size=100), random_seed=rng)
    )
    samples2 = pt.as_tensor(
        pm.draw(pm.Normal.dist(mu=mean2, sigma=std2, size=100), random_seed=rng)
    )

    # Calculate tail distances
    tail_distance_func = tail_distance(confidence_level=0.75)
    tail_distance1 = tail_distance_func(samples1, None).eval()
    tail_distance2 = tail_distance_func(samples2, None).eval()

    # Check that the tail distance is greater for the higher std deviation
    if expected_order == "greater":
        assert tail_distance2 > tail_distance1, (
            f"Expected tail distance to be greater for std={std2}, but got {tail_distance2} <= {tail_distance1}"
        )
    elif expected_order == "smaller":
        assert tail_distance1 > tail_distance2, (
            f"Expected tail distance to be greater for std={std1}, but got {tail_distance1} <= {tail_distance2}"
        )


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
def test_compare_mean_tightness_score(
    mean1, std1, mean2, std2, alpha, expected_relation
):
    # Generate samples for both distributions
    samples1 = pt.as_tensor(
        pm.draw(pm.Normal.dist(mu=mean1, sigma=std1, size=100), random_seed=rng)
    )
    samples2 = pt.as_tensor(
        pm.draw(pm.Normal.dist(mu=mean2, sigma=std2, size=100), random_seed=rng)
    )

    # Calculate mean tightness scores
    mean_tightness_score_func = mean_tightness_score(alpha=alpha, confidence_level=0.75)
    score1 = mean_tightness_score_func(samples1, None).eval()
    score2 = mean_tightness_score_func(samples2, None).eval()

    # Assertions based on observed behavior: higher mean should dominate in both cases
    if expected_relation == "higher_mean":
        assert score2 > score1, (
            f"Expected score for mean={mean2} to be higher, but got {score2} <= {score1}"
        )
    elif expected_relation == "lower_std":
        assert score1 > score2, (
            f"Expected score for std={std1} to be lower, but got {score1} <= {score2}"
        )


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


# Test Cases
@pytest.mark.parametrize(
    "data",
    [
        np.array([10, 20, 30, 40, 50]),  # Small dataset
        pm.draw(
            pm.Normal.dist(mu=10, sigma=5, size=50), random_seed=rng
        ),  # PyMC generated samples
        np.linspace(1, 100, 100),  # Linearly spaced values
        np.array([]),  # Empty array corner case
    ],
)
def test_compute_quantile(data):
    if data.size == 0:
        with pytest.raises(Exception, match=".*"):
            _compute_quantile(pt.as_tensor_variable(data), 0.95).eval()
    else:
        pytensor_quantile = _compute_quantile(pt.as_tensor_variable(data), 0.95).eval()
        numpy_quantile = np.quantile(data, 0.95)
        np.testing.assert_allclose(
            pytensor_quantile,
            numpy_quantile,
            rtol=1e-5,
            atol=1e-8,
            err_msg="Quantile mismatch",
        )


@pytest.mark.parametrize(
    "samples, budgets",
    [
        (
            pm.draw(pm.Normal.dist(mu=10, sigma=2, size=100), random_seed=rng),
            pm.draw(pm.Normal.dist(mu=300, sigma=50, size=100), random_seed=rng),
        ),
        (np.array([1, 2, 3]), np.array([100, 200, 300])),  # Simple case
    ],
)
def test_roas_distribution(samples, budgets):
    pt_samples = pt.as_tensor_variable(samples)
    pt_budgets = pt.as_tensor_variable(budgets)

    pytensor_roas = _calculate_roas_distribution_for_allocation(
        pt_samples, pt_budgets
    ).eval()
    numpy_roas = samples / np.sum(budgets)
    np.testing.assert_allclose(
        pytensor_roas, numpy_roas, rtol=1e-5, atol=1e-8, err_msg="ROAS mismatch"
    )


@pytest.mark.parametrize(
    "samples, budgets, func",
    [
        (
            pm.draw(pm.Normal.dist(mu=100, sigma=20, size=100), random_seed=rng),
            pm.draw(pm.Normal.dist(mu=1000, sigma=100, size=100), random_seed=rng),
            average_response,
        ),
        (
            pm.draw(pm.Normal.dist(mu=100, sigma=20, size=100), random_seed=rng),
            pm.draw(pm.Normal.dist(mu=1000, sigma=100, size=100), random_seed=rng),
            sharpe_ratio(0.01),
        ),
        (
            pm.draw(pm.Normal.dist(mu=100, sigma=20, size=100), random_seed=rng),
            pm.draw(pm.Normal.dist(mu=1000, sigma=100, size=100), random_seed=rng),
            raroc(0.01),
        ),
        (
            pm.draw(pm.Normal.dist(mu=100, sigma=20, size=100), random_seed=rng),
            pm.draw(pm.Normal.dist(mu=1000, sigma=100, size=100), random_seed=rng),
            portfolio_entropy,
        ),
    ],
)
def test_general_functions(samples, budgets, func):
    """
    Test utility functions for general behavior.
    """
    pt_samples = pt.as_tensor_variable(samples)
    pt_budgets = pt.as_tensor_variable(budgets)

    try:
        pytensor_result = func(pt_samples, pt_budgets).eval()
        assert pytensor_result is not None, "Function returned None"
    except Exception as e:
        pytest.fail(f"Function {func.__name__} raised an unexpected exception: {e!s}")


@pytest.mark.parametrize(
    "confidence_level",
    [
        0.0,
        1.0,
    ],
)
def test_value_at_risk_invalid_confidence_level(confidence_level, test_data):
    samples, budgets = test_data
    with pytest.raises(ValueError, match="Confidence level must be between 0 and 1."):
        value_at_risk(confidence_level)(samples, budgets).eval()


@pytest.mark.parametrize(
    "confidence_level",
    [
        0.0,
        1.0,
    ],
)
def test_conditional_value_at_risk_invalid_confidence_level(
    confidence_level, test_data
):
    samples, budgets = test_data
    with pytest.raises(ValueError, match="Confidence level must be between 0 and 1."):
        conditional_value_at_risk(confidence_level)(samples, budgets).eval()


@pytest.mark.parametrize(
    "confidence_level",
    [
        0.0,
        1.0,
    ],
)
def test_tail_distance_invalid_confidence_level(confidence_level, test_data):
    samples, budgets = test_data
    with pytest.raises(ValueError, match="Confidence level must be between 0 and 1."):
        tail_distance(confidence_level)(samples, budgets).eval()


@pytest.mark.parametrize(
    "confidence_level",
    [
        0.0,
        1.0,
    ],
)
def test_mean_tightness_score_invalid_confidence_level(confidence_level, test_data):
    samples, budgets = test_data
    with pytest.raises(ValueError, match="Confidence level must be between 0 and 1."):
        mean_tightness_score(alpha=0.5, confidence_level=confidence_level)(
            samples, budgets
        ).eval()


@pytest.mark.parametrize(
    "confidence_level",
    [
        0.0,
        1.0,
    ],
)
def test_adjusted_value_at_risk_score_invalid_confidence_level(
    confidence_level, test_data
):
    samples, budgets = test_data
    with pytest.raises(ValueError, match="Confidence level must be between 0 and 1."):
        adjusted_value_at_risk_score(
            confidence_level=confidence_level, risk_aversion=0.8
        )(samples, budgets).eval()


@pytest.mark.parametrize(
    "risk_aversion",
    [
        -0.1,
        1.1,
    ],
)
def test_adjusted_value_at_risk_score_invalid_risk_aversion(risk_aversion, test_data):
    samples, budgets = test_data
    with pytest.raises(
        ValueError, match="Risk aversion parameter must be between 0 and 1."
    ):
        adjusted_value_at_risk_score(
            confidence_level=0.95, risk_aversion=risk_aversion
        )(samples, budgets).eval()
