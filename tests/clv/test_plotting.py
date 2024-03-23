import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytest

from pymc_marketing.clv.models import BetaGeoModel, ParetoNBDModel
from pymc_marketing.clv.plotting import (
    plot_customer_exposure,
    plot_frequency_recency_matrix,
    plot_probability_alive_matrix,
)
from tests.conftest import set_model_fit


@pytest.fixture(scope="module")
def fitted_bg(test_summary_data) -> BetaGeoModel:
    rng = np.random.default_rng(13)
    data = pd.DataFrame(
        {
            "customer_id": test_summary_data.index,
            "frequency": test_summary_data["frequency"],
            "recency": test_summary_data["recency"],
            "T": test_summary_data["T"],
        }
    )
    model_config = {
        # Narrow Gaussian centered at MLE params from lifetimes BetaGeoFitter
        "a_prior": {"dist": "DiracDelta", "kwargs": {"c": 1.85034151}},
        "alpha_prior": {"dist": "DiracDelta", "kwargs": {"c": 1.86428187}},
        "b_prior": {"dist": "DiracDelta", "kwargs": {"c": 3.18105431}},
        "r_prior": {"dist": "DiracDelta", "kwargs": {"c": 0.16385072}},
    }
    model = BetaGeoModel(
        data=data,
        model_config=model_config,
    )
    model.build_model()
    fake_fit = pm.sample_prior_predictive(
        samples=50, model=model.model, random_seed=rng
    ).prior
    set_model_fit(model, fake_fit)

    return model


@pytest.fixture(scope="module")
def fitted_pnbd(test_summary_data) -> ParetoNBDModel:
    rng = np.random.default_rng(45)

    model_config = {
        # Narrow Gaussian centered at MLE params from lifetimes ParetoNBDFitter
        "r_prior": {"dist": "DiracDelta", "kwargs": {"c": 0.5534}},
        "alpha_prior": {"dist": "DiracDelta", "kwargs": {"c": 10.5802}},
        "s_prior": {"dist": "DiracDelta", "kwargs": {"c": 0.6061}},
        "beta_prior": {"dist": "DiracDelta", "kwargs": {"c": 11.6562}},
    }
    pnbd_model = ParetoNBDModel(
        data=test_summary_data,
        model_config=model_config,
    )
    pnbd_model.build_model()

    # Mock an idata object for tests requiring a fitted model
    # TODO: This is quite slow. Check similar fixtures in the model tests to speed this up.
    fake_fit = pm.sample_prior_predictive(
        samples=50, model=pnbd_model.model, random_seed=rng
    ).prior
    set_model_fit(pnbd_model, fake_fit)

    return pnbd_model


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


def test_plot_cumstomer_exposure_with_ax(test_summary_data) -> None:
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


@pytest.mark.parametrize("mock_model", (fitted_bg, fitted_pnbd))
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
