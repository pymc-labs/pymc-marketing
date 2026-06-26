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
import pytest
import xarray as xr
from pymc_extras.prior import Prior

from pymc_marketing.bass import BassModel

PLOT_METHODS = [
    "plot_adoption_curve",
    "plot_cumulative",
    "plot_decomposition",
    "plot_peak",
]


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def single_product_model(mock_pymc_sample) -> BassModel:
    y = np.random.default_rng(42).poisson(lam=100, size=20)
    model = BassModel()
    model.fit(data=y, draws=20, tune=5, chains=1, random_seed=42)
    return model


@pytest.fixture(scope="module")
def multi_product_model(mock_pymc_sample) -> BassModel:
    products = ["A", "B", "C"]
    counts = np.random.default_rng(42).poisson(lam=100, size=(20, len(products)))
    ds = xr.Dataset(
        {"observed": (("T", "product"), counts)},
        coords={"T": np.arange(20), "product": products},
    )
    model = BassModel(
        model_config={
            "m": Prior("Normal", mu=1000, sigma=200, dims="product"),
            "p": Prior("Beta", alpha=1.5, beta=20, dims="product"),
            "q": Prior("Beta", alpha=2, beta=5, dims="product"),
            "likelihood": Prior("Poisson"),
        },
    )
    model.fit(data=ds, draws=20, tune=5, chains=1, random_seed=42)
    return model


@pytest.mark.parametrize("method", PLOT_METHODS)
def test_single_product(single_product_model: BassModel, method: str) -> None:
    fig, axes = getattr(single_product_model, method)()

    assert isinstance(fig, plt.Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.size == 1


@pytest.mark.parametrize("method", PLOT_METHODS)
def test_multi_product_grid(multi_product_model: BassModel, method: str) -> None:
    fig, axes = getattr(multi_product_model, method)()

    assert isinstance(fig, plt.Figure)
    assert axes.size == 3


@pytest.mark.parametrize("method", PLOT_METHODS)
def test_multi_product_selection(multi_product_model: BassModel, method: str) -> None:
    fig, axes = getattr(multi_product_model, method)(product="B")

    assert isinstance(fig, plt.Figure)
    assert axes.size == 1


@pytest.mark.parametrize("n_products", [1, 3], ids=["single", "multi"])
def test_decomposition_twin_axes(
    single_product_model: BassModel,
    multi_product_model: BassModel,
    n_products: int,
) -> None:
    model = single_product_model if n_products == 1 else multi_product_model
    fig, axes = model.plot_decomposition()

    # One twin (right) axis per primary (left) axis
    assert axes.size == n_products
    assert len(fig.axes) == 2 * n_products


def test_observed_data_overlaid(single_product_model: BassModel) -> None:
    _, axes = single_product_model.plot_adoption_curve()

    observed = single_product_model.idata.fit_data["observed"].values
    overlay = [line for line in axes.flat[0].get_lines() if line.get_color() == "black"]
    assert len(overlay) == 1
    np.testing.assert_array_equal(overlay[0].get_ydata(), observed)


def test_observed_data_aligned_per_product(multi_product_model: BassModel) -> None:
    _, axes = multi_product_model.plot_adoption_curve()

    observed = multi_product_model.idata.fit_data["observed"]
    products = observed.coords["product"].values
    for ax, product in zip(axes.flat, products, strict=True):
        overlay = [line for line in ax.get_lines() if line.get_color() == "black"]
        assert len(overlay) == 1
        np.testing.assert_array_equal(
            overlay[0].get_ydata(), observed.sel(product=product).values
        )


@pytest.mark.parametrize("method", PLOT_METHODS)
def test_unfitted_model_raises(method: str) -> None:
    model = BassModel()
    with pytest.raises(RuntimeError, match="hasn't been fit"):
        getattr(model, method)()


@pytest.mark.parametrize("method", PLOT_METHODS)
def test_product_on_single_product_raises(
    single_product_model: BassModel, method: str
) -> None:
    with pytest.raises(ValueError, match="no 'product' dimension"):
        getattr(single_product_model, method)(product="A")
