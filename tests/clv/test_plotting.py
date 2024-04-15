from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytensor.tensor import TensorVariable

from pymc_marketing.clv.plotting import (
    plot_customer_exposure,
    plot_frequency_recency_matrix,
    plot_probability_alive_matrix,
)


class MockModel:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def _mock_posterior(
        self, customer_id: Union[np.ndarray, pd.Series]
    ) -> xr.DataArray:
        n_customers = len(customer_id)
        n_chains = 4
        n_draws = 10
        chains = np.arange(n_chains)
        draws = np.arange(n_draws)
        return xr.DataArray(
            data=np.ones((n_customers, n_chains, n_draws)),
            coords={"customer_id": customer_id, "chain": chains, "draw": draws},
            dims=["customer_id", "chain", "draw"],
        )

    def expected_probability_alive(
        self,
        customer_id: Union[np.ndarray, pd.Series],
        frequency: Union[np.ndarray, pd.Series],
        recency: Union[np.ndarray, pd.Series],
        T: Union[np.ndarray, pd.Series],
    ):
        return self._mock_posterior(customer_id)

    def expected_purchases(
        self,
        customer_id: Union[np.ndarray, pd.Series],
        data: pd.DataFrame,
        *,
        future_t: Union[np.ndarray, pd.Series, TensorVariable],
    ):
        return self._mock_posterior(customer_id)

    # TODO: This is required until CLV API is standardized.
    def expected_num_purchases(
        self,
        customer_id: Union[np.ndarray, pd.Series],
        t: Union[np.ndarray, pd.Series, TensorVariable],
        frequency: Union[np.ndarray, pd.Series, TensorVariable],
        recency: Union[np.ndarray, pd.Series, TensorVariable],
        T: Union[np.ndarray, pd.Series, TensorVariable],
    ):
        return self._mock_posterior(customer_id)


@pytest.fixture
def mock_model(test_summary_data) -> MockModel:
    return MockModel(test_summary_data)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"colors": ["blue", "red"]},
        {"labels": ["Customer Recency", "Customer T"]},
    ],
)
def test_plot_customer_exposure(test_summary_data, kwargs) -> None:
    ax: plt.Axes = plot_customer_exposure(test_summary_data, **kwargs)

    assert isinstance(ax, plt.Axes)


def test_plot_customer_exposure_with_ax(test_summary_data) -> None:
    ax = plt.subplot()
    plot_customer_exposure(test_summary_data, ax=ax)

    assert ax.get_title() == "Customer Exposure"
    assert ax.get_xlabel() == "Time since first purchase"
    assert ax.get_ylabel() == "Customer"


@pytest.mark.parametrize(
    "kwargs",
    [
        # More labels or colors
        {"labels": [0, 1, 2]},
        {"colors": ["blue", "red", "green"]},
        # Negative Values
        {"padding": -1},
        {"linewidth": -1},
        {"size": -1},
    ],
)
def test_plot_customer_exposure_invalid_args(test_summary_data, kwargs) -> None:
    with pytest.raises(ValueError):
        plot_customer_exposure(test_summary_data, **kwargs)


def test_plot_frequency_recency_matrix(mock_model) -> None:
    ax: plt.Axes = plot_frequency_recency_matrix(mock_model)

    assert isinstance(ax, plt.Axes)


def test_plot_frequency_recency_matrix_bounds(mock_model) -> None:
    max_recency = 10
    max_frequency = 10
    ax: plt.Axes = plot_frequency_recency_matrix(
        mock_model, max_recency=max_recency, max_frequency=max_frequency
    )

    assert isinstance(ax, plt.Axes)


def test_plot_frequency_recency_matrix_with_ax(mock_model) -> None:
    ax = plt.subplot()
    plot_frequency_recency_matrix(mock_model, ax=ax)

    assert ax.get_xlabel() == "Customer's Historical Frequency"
    assert ax.get_ylabel() == "Customer's Recency"


def test_plot_probability_alive_matrix(mock_model) -> None:
    ax: plt.Axes = plot_probability_alive_matrix(mock_model)

    assert isinstance(ax, plt.Axes)


def test_plot_probability_alive_matrix_bounds(mock_model) -> None:
    max_recency = 10
    max_frequency = 10
    ax: plt.Axes = plot_probability_alive_matrix(
        mock_model, max_recency=max_recency, max_frequency=max_frequency
    )

    assert isinstance(ax, plt.Axes)


def test_plot_probability_alive_matrix_with_ax(mock_model) -> None:
    ax = plt.subplot()
    plot_probability_alive_matrix(mock_model, ax=ax)

    assert ax.get_xlabel() == "Customer's Historical Frequency"
    assert ax.get_ylabel() == "Customer's Recency"
