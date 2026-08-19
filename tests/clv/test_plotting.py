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
from pymc_marketing.clv.plotting import _plot_expected_purchases_ecdf


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


def test_plot_expected_purchases_ppc_exceptions(fitted_model):
    with pytest.raises(
        NameError, match=r"Specify 'prior' or 'posterior' for 'ppc' parameter."
    ):
        plot_expected_purchases_ppc(fitted_model, ppc="ppc")

    with pytest.raises(
        ValueError, match=r"Specify 'hist' or 'ecdf' for 'plot_type' parameter."
    ):
        plot_expected_purchases_ppc(fitted_model, plot_type="bar")

    with pytest.raises(ValueError, match=r"sequence of two matplotlib Axes"):
        plot_expected_purchases_ppc(
            fitted_model, plot_type="ecdf", ax=plt.subplots(1, 1)[1]
        )

    with pytest.raises(ValueError, match=r"single matplotlib Axes"):
        plot_expected_purchases_ppc(fitted_model, ax=plt.subplots(2, 1)[1])

    with pytest.raises(ValueError, match=r"distinct observed purchase count"):
        _plot_expected_purchases_ecdf(
            observed=np.ones(10),
            ppc=np.ones(10),
            title_prefix="Posterior Predictive",
            random_seed=45,
        )

    plt.close("all")


@pytest.mark.parametrize(
    "ppc, max_purchases, samples, use_ax",
    [("prior", 10, 100, False), ("posterior", 20, 50, True)],
)
def test_plot_expected_purchases_ppc(fitted_model, ppc, max_purchases, samples, use_ax):
    subplot = plt.subplots(1, 1)[1] if use_ax else None
    ax = plot_expected_purchases_ppc(
        model=fitted_model,
        ppc=ppc,
        max_purchases=max_purchases,
        samples=samples,
        ax=subplot,
    )

    # the default plot_type is 'hist', which returns a single Axes
    assert isinstance(ax, plt.Axes)
    if use_ax:
        assert ax is subplot

    # clear any existing pyplot figures
    plt.close("all")


@pytest.mark.parametrize("ppc", ["prior", "posterior"])
@pytest.mark.parametrize("use_ax", [False, True])
def test_plot_expected_purchases_ppc_ecdf(fitted_model, ppc, use_ax):
    subplots = plt.subplots(2, 1)[1] if use_ax else None
    ax_ecdf, ax_diff = plot_expected_purchases_ppc(
        model=fitted_model,
        ppc=ppc,
        plot_type="ecdf",
        samples=100,
        ax=subplots,
    )

    assert isinstance(ax_ecdf, plt.Axes)
    assert isinstance(ax_diff, plt.Axes)
    if use_ax:
        assert (ax_ecdf, ax_diff) == tuple(subplots)
    else:
        assert ax_ecdf.figure is ax_diff.figure

    ecdf_y = ax_ecdf.lines[0].get_ydata()
    assert np.all(np.diff(ecdf_y) >= 0)
    assert ecdf_y.min() >= 0.0
    assert ecdf_y.max() <= 1.0

    # the confidence band stays within [0, 1]
    band_y = ax_ecdf.collections[0].get_paths()[0].vertices[:, 1]
    assert band_y.min() >= 0.0
    assert band_y.max() <= 1.0

    # the difference panel is drawn on the same grid as the ECDF panel
    assert np.array_equal(ax_diff.lines[0].get_xdata(), ax_ecdf.lines[0].get_xdata())

    # clear any existing pyplot figures
    plt.close("all")


def test_plot_expected_purchases_ppc_ecdf_ignores_max_purchases(fitted_model):
    with pytest.warns(UserWarning, match=r"'max_purchases' is ignored"):
        plot_expected_purchases_ppc(
            model=fitted_model,
            plot_type="ecdf",
            max_purchases=20,
        )

    plt.close("all")
