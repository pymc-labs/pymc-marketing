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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pytensor.tensor import TensorVariable

from pymc_marketing.clv import (
    plot_customer_exposure,
    plot_expected_purchases_over_time,
    plot_expected_purchases_ppc,
    plot_frequency_recency_matrix,
    plot_probability_alive_matrix,
)


class MockModel:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self._model_type = None

    def _mock_posterior(self, data: pd.DataFrame) -> xr.DataArray:
        n_customers = len(data)
        n_chains = 4
        n_draws = 10
        chains = np.arange(n_chains)
        draws = np.arange(n_draws)
        return xr.DataArray(
            data=np.ones((n_customers, n_chains, n_draws)),
            coords={"customer_id": data["customer_id"], "chain": chains, "draw": draws},
            dims=["customer_id", "chain", "draw"],
        )

    def expected_probability_alive(self, data: np.ndarray | pd.Series):
        return self._mock_posterior(data)

    def expected_purchases(
        self,
        data: pd.DataFrame,
        *,
        future_t: np.ndarray | pd.Series | TensorVariable,
    ):
        return self._mock_posterior(data)

    def expected_purchases_new_customer(
        self,
        data: pd.DataFrame,
    ):
        return self._mock_posterior(data)


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


@pytest.mark.parametrize(
    "plot_cumulative, set_index_date, subplot",
    [(True, False, None), (False, True, plt.subplot())],
)
def test_plot_expected_purchases_over_time(
    mock_model, cdnow_trans, plot_cumulative, set_index_date, subplot
) -> None:
    ax = plot_expected_purchases_over_time(
        model=mock_model,
        purchase_history=cdnow_trans,
        customer_id_col="id",
        datetime_col="date",
        datetime_format="%Y%m%d",
        time_unit="D",
        plot_cumulative=plot_cumulative,
        set_index_date=set_index_date,
        t=10,
        t_start_eval=8,
        ax=subplot,
    )

    assert isinstance(ax, plt.Axes)

    # clear any existing pyplot figures
    plt.clf()


def test_plot_expected_purchases_over_time_exceptions(mock_model, cdnow_trans):
    with pytest.warns(
        DeprecationWarning,
        match="t_unobserved is deprecated and will be removed in a future release. "
        "Use t_start_eval instead.",
    ):
        plot_expected_purchases_over_time(
            model=mock_model,
            purchase_history=cdnow_trans,
            customer_id_col="id",
            datetime_col="date",
            datetime_format="%Y%m%d",
            time_unit="D",
            t=10,
            t_unobserved=8,
        )

    # clear any existing pyplot figures
    plt.clf()


def test_plot_expected_purchases_ppc_exceptions(fitted_model):
    with pytest.raises(
        NameError, match="Specify 'prior' or 'posterior' for 'ppc' parameter."
    ):
        plot_expected_purchases_ppc(fitted_model, ppc="ppc")


@pytest.mark.parametrize(
    "ppc, max_purchases, samples, subplot",
    [("prior", 10, 100, None), ("posterior", 20, 50, plt.subplot())],
)
def test_plot_expected_purchases_ppc(
    fitted_model, ppc, max_purchases, samples, subplot
):
    ax = plot_expected_purchases_ppc(
        model=fitted_model,
        ppc=ppc,
        max_purchases=max_purchases,
        samples=samples,
        ax=subplot,
    )

    assert isinstance(ax, plt.Axes)

    # clear any existing pyplot figures
    plt.clf()
