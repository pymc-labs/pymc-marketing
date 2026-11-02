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
"""Behavioral contracts of the experimental SCM: fitting, prior checks, and forecasting."""

from collections.abc import Callable
from copy import copy
from typing import Any

import numpy as np
import pandas as pd
import pymc.dims as pmd
import pytest
import xarray as xr
from pymc_extras.prior import Prior
from scipy.special import gammaln
from scipy.stats import norm

from pymc_marketing.experimental.terms import SCM, Data, Equation
from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.terms import Dot, Intercept, Parameter, Transform

SAMPLE_KWARGS = {
    "draws": 30,
    "tune": 40,
    "chains": 1,
    "cores": 1,
    "random_seed": 892,
    "progressbar": False,
    "compute_convergence_checks": False,
}
PREDICT_KWARGS = {"random_seed": 7, "progressbar": False}

ALPHA = 0.4
LAM = 1.1
L_MAX = 3
CHANNELS = ["tv", "radio"]
PRODUCTS = ["basic", "premium"]
TARGETS = ["units", "revenue"]
TRAIN_DATES = pd.date_range("2025-01-06", periods=12, freq="W-MON")
FUTURE_DATES = pd.date_range(TRAIN_DATES[-1], periods=4, freq="W-MON")[1:]


def _adstock(spend, alpha, l_max):
    """Normalized geometric adstock along axis 0 with zeros before the first row."""
    weights = alpha ** np.arange(l_max)
    weights = weights / weights.sum()
    out = np.zeros_like(spend, dtype=float)
    for lag, weight in enumerate(weights):
        out[lag:] += weight * spend[: spend.shape[0] - lag]
    return out


def _sales_mean_oracle(posterior, spend, price, history):
    """Mean sales (chain, draw, date, product, target) from raw spend and price arrays.

    ``spend`` is (date, channel) including ``history`` leading rows that are dropped
    after adstock; ``price`` is (date, product) for the returned dates only.
    """
    adstocked = _adstock(spend, ALPHA, L_MAX)[history:]
    beta = posterior["saturation_beta"].transpose("chain", "draw", "channel", "target")
    saturated = beta.values[:, :, None, :, :] * np.tanh(
        LAM * adstocked[None, None, :, :, None] / 2
    )
    response = saturated.sum(axis=3)
    intercept = posterior["intercept"].transpose("chain", "draw", "product", "target")
    price_beta = posterior["price_beta"].transpose("chain", "draw", "target")
    return (
        intercept.values[:, :, None, :, :]
        + response[:, :, :, None, :]
        + price[None, None, :, :, None] * price_beta.values[:, :, None, None, :]
    )


def _multi_target_training():
    rng = np.random.default_rng(11)
    spend = rng.gamma(2.0, 1.0, size=(len(TRAIN_DATES), len(CHANNELS)))
    price = rng.uniform(1.0, 3.0, size=(len(TRAIN_DATES), len(PRODUCTS)))
    sales = 5.0 + rng.normal(size=(len(TRAIN_DATES), len(PRODUCTS), len(TARGETS)))
    return xr.Dataset(
        {
            "spend": (("date", "channel"), spend),
            "price": (("date", "product"), price),
            "sales": (("date", "product", "target"), sales),
        },
        coords={
            "date": TRAIN_DATES,
            "channel": CHANNELS,
            "product": PRODUCTS,
            "target": TARGETS,
        },
    )


def _multi_target_future(spend, price, *, channel=CHANNELS, coords=None):
    return xr.Dataset(
        {
            "spend": (("date", "channel"), spend),
            "price": (("date", "product"), price),
        },
        coords={
            "date": FUTURE_DATES,
            "channel": channel,
            "product": PRODUCTS,
            **(coords or {}),
        },
    )


def _multi_target_recipe():
    response = (
        Data("spend")
        >> GeometricAdstock(l_max=L_MAX, priors={"alpha": ALPHA})
        >> LogisticSaturation(
            priors={
                "lam": LAM,
                "beta": Prior("HalfNormal", sigma=1, dims=("channel", "target")),
            }
        )
    )
    mean = (
        Intercept(prior=Prior("Normal", dims=("product", "target")))
        + response.sum("channel")
        + Data("price") * Parameter("price_beta", Prior("Normal", dims="target"))
    )
    return Equation(
        observed="sales",
        mu=mean.named("sales_mean", dims=("date", "product", "target")),
        likelihood=Prior("Normal", sigma=Prior("HalfNormal", sigma=1)),
    )


@pytest.fixture(scope="module")
def multi_target():
    train = _multi_target_training()
    rng = np.random.default_rng(23)
    future = _multi_target_future(
        rng.gamma(2.0, 1.0, size=(len(FUTURE_DATES), len(CHANNELS))),
        rng.uniform(1.0, 3.0, size=(len(FUTURE_DATES), len(PRODUCTS))),
    )
    scm = SCM(_multi_target_recipe())
    scm.fit(train, **SAMPLE_KWARGS)
    return scm, train, future


def _forecast_mean(scm, future, **kwargs):
    return scm.sample_posterior_predictive(
        future, var_names=["sales", "sales_mean"], **PREDICT_KWARGS, **kwargs
    )


def _oracle_with_history(scm, train, future):
    spend = np.concatenate(
        [train["spend"].values[-(L_MAX - 1) :], future["spend"].values]
    )
    return _sales_mean_oracle(
        scm.idata["posterior"].to_dataset(), spend, future["price"].values, L_MAX - 1
    )


def test_multi_target_forecast_matches_numpy_oracle(multi_target):
    scm, train, future = multi_target
    prediction = _forecast_mean(scm, future)

    assert prediction["sales"].sizes == {
        "chain": 1,
        "draw": SAMPLE_KWARGS["draws"],
        "date": len(FUTURE_DATES),
        "product": len(PRODUCTS),
        "target": len(TARGETS),
    }
    assert prediction["sales_mean"].sizes == prediction["sales"].sizes
    np.testing.assert_array_equal(prediction["date"].values, FUTURE_DATES.values)
    np.testing.assert_allclose(
        prediction["sales_mean"]
        .transpose("chain", "draw", "date", "product", "target")
        .values,
        _oracle_with_history(scm, train, future),
        atol=1e-10,
    )


def test_forecast_is_invariant_to_label_order(multi_target):
    scm, _, future = multi_target
    reversed_labels = _multi_target_future(
        future["spend"].values[:, ::-1],
        future["price"].values,
        channel=CHANNELS[::-1],
        coords={"target": TARGETS[::-1]},
    )

    baseline = _forecast_mean(scm, future)["sales_mean"]
    reordered = _forecast_mean(scm, reversed_labels)["sales_mean"]

    assert list(reordered["target"].values) == TARGETS
    np.testing.assert_allclose(reordered.values, baseline.values, atol=1e-10)


def test_forecast_responds_to_future_spend(multi_target):
    scm, train, future = multi_target
    scaled = future.assign(spend=future["spend"] * 1.7)

    baseline = _forecast_mean(scm, future)["sales_mean"]
    prediction = _forecast_mean(scm, scaled)["sales_mean"]

    assert not np.allclose(prediction.values, baseline.values)
    np.testing.assert_allclose(
        prediction.transpose("chain", "draw", "date", "product", "target").values,
        _oracle_with_history(scm, train, scaled),
        atol=1e-10,
    )


def test_forecast_without_history_pads_with_zeros(multi_target):
    scm, _, future = multi_target

    with_history = _forecast_mean(scm, future)["sales_mean"]
    scenario = _forecast_mean(scm, future, include_last_observations=False)[
        "sales_mean"
    ]

    assert not np.allclose(scenario.values, with_history.values)
    np.testing.assert_allclose(
        scenario.transpose("chain", "draw", "date", "product", "target").values,
        _sales_mean_oracle(
            scm.idata["posterior"].to_dataset(),
            future["spend"].values,
            future["price"].values,
            0,
        ),
        atol=1e-10,
    )


@pytest.mark.parametrize(
    ("alter", "error", "match"),
    [
        pytest.param(
            lambda future: future.assign_coords(
                date=future["date"] + pd.Timedelta(weeks=1)
            ),
            ValueError,
            "immediately follow training",
            id="shifted-dates",
        ),
        pytest.param(
            lambda future: future.assign_coords(target=TARGETS[:1]),
            ValueError,
            "labels for 'target' must match training",
            id="target-subset",
        ),
        pytest.param(
            lambda future: future.drop_vars("channel"),
            ValueError,
            "'channel' must provide coordinate labels",
            id="unlabeled-channel",
        ),
        pytest.param(
            lambda future: future["price"],
            TypeError,
            "xarray.Dataset",
            id="dataarray-instead-of-dataset",
        ),
        pytest.param(
            lambda future: future.assign_coords(
                target=("date", [TARGETS[0]] * future.sizes["date"])
            ),
            ValueError,
            "'target' must label its own dimension",
            id="coordinate-on-wrong-dimension",
        ),
        pytest.param(
            lambda future: future.rename(date="period"),
            ValueError,
            "requires a date dimension",
            id="missing-date-dimension",
        ),
    ],
)
def test_forecast_rejects_inconsistent_future_data(
    multi_target: tuple[SCM, xr.Dataset, xr.Dataset],
    alter: Callable[[xr.Dataset], Any],
    error: type[Exception],
    match: str,
) -> None:
    scm, _, future = multi_target
    fitted_model, fitted_idata = scm.model, scm.idata
    future_before = future.copy(deep=True)
    posterior_before = scm.idata["posterior"].to_dataset().copy(deep=True)
    with pytest.raises(error, match=match):
        _forecast_mean(scm, alter(future))
    assert scm.model is fitted_model
    assert scm.idata is fitted_idata
    xr.testing.assert_identical(future, future_before)
    xr.testing.assert_identical(scm.idata["posterior"].to_dataset(), posterior_before)


def test_build_model_requires_every_observation_variable():
    train = _multi_target_training().drop_vars("sales")
    with pytest.raises(ValueError, match="missing observations 'sales'"):
        SCM(_multi_target_recipe()).build_model(train)


def test_forecast_leaves_fit_untouched(multi_target):
    scm, train, future = multi_target
    fitted_model = scm.model
    train_before = train.copy(deep=True)
    posterior_before = scm.idata["posterior"].to_dataset().copy(deep=True)

    _forecast_mean(scm, future.isel(channel=slice(None, None, -1)))

    assert scm.model is fitted_model
    xr.testing.assert_identical(train, train_before)
    xr.testing.assert_identical(scm.idata["posterior"].to_dataset(), posterior_before)


def test_prior_predictive_samples_every_equation_without_touching_the_fit(
    multi_target,
):
    scm, train, _ = multi_target
    fitted_model = scm.model
    posterior_before = scm.idata["posterior"].to_dataset().copy(deep=True)

    prior = scm.sample_prior_predictive(train, draws=25, random_seed=3)

    assert sorted(prior.children) == ["observed_data", "prior", "prior_predictive"]
    assert prior["prior_predictive"]["sales"].sizes == {
        "chain": 1,
        "draw": 25,
        "date": len(TRAIN_DATES),
        "product": len(PRODUCTS),
        "target": len(TARGETS),
    }
    assert (
        prior["prior"]["sales_mean"].sizes == prior["prior_predictive"]["sales"].sizes
    )
    assert scm.model is fitted_model
    xr.testing.assert_identical(scm.idata["posterior"].to_dataset(), posterior_before)


def test_prior_predictive_requires_observations():
    train = _multi_target_training().drop_vars("sales")
    with pytest.raises(ValueError, match="missing observations 'sales'"):
        SCM(_multi_target_recipe()).sample_prior_predictive(train, draws=5)


def test_prior_sampling_rejects_managed_arguments_without_changing_fit(
    multi_target: tuple[SCM, xr.Dataset, xr.Dataset],
) -> None:
    scm, train, _ = multi_target
    fitted_model, fitted_idata = scm.model, scm.idata
    posterior_before = scm.idata["posterior"].to_dataset().copy(deep=True)

    with pytest.raises(ValueError, match="manages"):
        scm.sample_prior_predictive(train, draws=1, return_inferencedata=False)

    assert scm.model is fitted_model
    assert scm.idata is fitted_idata
    xr.testing.assert_identical(scm.idata["posterior"].to_dataset(), posterior_before)


DRIVER = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
REVENUE = np.array([4.1, 6.0, 7.9, 10.2, 12.1])
ORDERS = np.array([2, 3, 3, 5, 6])
LIFT = np.array([3.6, 4.2, 3.9])


def _shared_slope_training():
    return xr.Dataset(
        {
            "driver": ("date", DRIVER),
            "revenue_obs": ("date", REVENUE),
            "orders_obs": ("date", ORDERS),
            "lift": ("study", LIFT),
        },
        coords={
            "date": pd.date_range("2025-03-01", periods=len(DRIVER), freq="D"),
            "study": ["geo_a", "geo_b", "geo_c"],
        },
    )


@pytest.fixture(scope="module")
def shared_slope():
    slope = Parameter("slope", Prior("Normal", mu=4, sigma=1))
    revenue = Equation(
        name="revenue",
        observed="revenue_obs",
        mu=(2.0 + slope * Data("driver")).named("revenue_mean"),
        likelihood=Prior("Normal", sigma=0.1),
    )
    orders = Equation(
        name="orders",
        observed="orders_obs",
        mu=Transform(0.3 + 0.2 * (slope * Data("driver")), pmd.math.exp).named(
            "order_rate"
        ),
        likelihood=Prior("Poisson"),
    )
    calibration = Equation(
        name="calibration",
        observed="lift",
        mu=slope,
        likelihood=Prior("Normal", sigma=0.2),
    )
    scm = SCM(revenue, orders, calibration)
    scm.fit(_shared_slope_training(), **SAMPLE_KWARGS)
    return scm


def _shared_slope_logp(slope):
    rate = np.exp(0.3 + 0.2 * slope * DRIVER)
    return (
        norm.logpdf(slope, loc=4, scale=1)
        + norm.logpdf(REVENUE, loc=2.0 + slope * DRIVER, scale=0.1).sum()
        + (ORDERS * np.log(rate) - rate - gammaln(ORDERS + 1)).sum()
        + norm.logpdf(LIFT, loc=slope, scale=0.2).sum()
    )


@pytest.mark.parametrize("slope", [3.3, 4.0, 4.8])
def test_shared_parameter_joins_three_likelihood_families(shared_slope, slope):
    model = shared_slope.model

    assert [rv.name for rv in model.free_RVs] == ["slope"]
    assert len(model.observed_RVs) == 3
    np.testing.assert_allclose(
        model.compile_logp()({"slope": slope}), _shared_slope_logp(slope), atol=1e-8
    )


def test_shared_parameter_prediction_uses_posterior_draws(shared_slope):
    driver = np.array([3.0, 3.5, 4.0])
    future = xr.Dataset(
        {"driver": ("date", driver)},
        coords={"date": pd.date_range("2025-03-06", periods=3, freq="D")},
    )
    slope = shared_slope.idata["posterior"]["slope"].transpose("chain", "draw").values

    prediction = shared_slope.sample_posterior_predictive(
        future,
        var_names=["revenue_mean", "order_rate", "orders", "calibration"],
        **PREDICT_KWARGS,
    )

    expected_mean = 2.0 + slope[:, :, None] * driver
    np.testing.assert_allclose(
        prediction["revenue_mean"].transpose("chain", "draw", "date").values,
        expected_mean,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        prediction["order_rate"].transpose("chain", "draw", "date").values,
        np.exp(0.3 + 0.2 * slope[:, :, None] * driver),
        atol=1e-10,
    )
    orders = prediction["orders"].values
    assert np.issubdtype(orders.dtype, np.integer)
    assert (orders >= 0).all()
    assert prediction["calibration"].dims == ("chain", "draw", "study")
    assert list(prediction["study"].values) == ["geo_a", "geo_b", "geo_c"]


TV = np.array([1.0, 0.5, 2.0, 1.5, 0.0, 1.0, 2.5, 0.5])


def _conditioning_training():
    rng = np.random.default_rng(5)
    search = 2.0 * TV + rng.normal(scale=0.1, size=TV.size)
    return xr.Dataset(
        {
            "tv": ("date", TV),
            "search": ("date", search),
            "sales": ("date", 1.5 * search + rng.normal(scale=0.2, size=TV.size)),
        },
        coords={"date": pd.date_range("2025-06-01", periods=TV.size, freq="D")},
    )


@pytest.fixture(scope="module")
def conditioning():
    search = Equation(
        observed="search",
        mu=2 * Data("tv"),
        likelihood=Prior("Normal", sigma=0.1),
    )
    sales = Equation(
        observed="sales",
        mu=Parameter("beta", Prior("Normal")) * search,
        likelihood=Prior("Normal", sigma=Prior("HalfNormal")),
    )
    scm = SCM(search, sales)
    scm.fit(_conditioning_training(), **SAMPLE_KWARGS)
    future = xr.Dataset(
        {
            "tv": ("date", np.array([1.0, 2.0, 3.0])),
            "search": ("date", np.array([2.2, 3.9, 6.1])),
        },
        coords={"date": pd.date_range("2025-06-09", periods=3, freq="D")},
    )
    return scm, future


def test_condition_on_holds_supplied_observations_fixed(conditioning):
    scm, future = conditioning

    conditioned = scm.sample_posterior_predictive(
        future.drop_vars("tv"), condition_on=["search"], **PREDICT_KWARGS
    )
    generated = scm.sample_posterior_predictive(
        future, condition_on=(), **PREDICT_KWARGS
    )

    held = conditioned["search"].transpose("chain", "draw", "date").values
    np.testing.assert_array_equal(
        held, np.broadcast_to(future["search"].values, held.shape)
    )
    assert (generated["search"].std(dim="draw") > 0).all()


@pytest.mark.parametrize(
    ("condition_on", "error"),
    [
        pytest.param("search", TypeError, id="bare-string"),
        pytest.param(["clicks"], ValueError, id="unknown-name"),
        pytest.param([42], TypeError, id="nonstring-selector"),
    ],
)
def test_condition_on_rejects_invalid_selectors(conditioning, condition_on, error):
    scm, future = conditioning
    with pytest.raises(error):
        scm.sample_posterior_predictive(
            future, condition_on=condition_on, **PREDICT_KWARGS
        )


def test_condition_on_requires_the_selected_observations(
    conditioning: tuple[SCM, xr.Dataset],
) -> None:
    scm, future = conditioning
    fitted_model, fitted_idata = scm.model, scm.idata
    with pytest.raises(ValueError, match="Required observations 'search'"):
        scm.sample_posterior_predictive(
            future.drop_vars("search"), condition_on=["search"], **PREDICT_KWARGS
        )
    assert scm.model is fitted_model
    assert scm.idata is fitted_idata


def test_prediction_rejects_managed_arguments_without_changing_fit(
    conditioning: tuple[SCM, xr.Dataset],
) -> None:
    scm, future = conditioning
    fitted_model, fitted_idata = scm.model, scm.idata
    posterior_before = scm.idata["posterior"].to_dataset().copy(deep=True)

    with pytest.raises(ValueError, match="manages"):
        scm.sample_posterior_predictive(future, freeze_vars=[], **PREDICT_KWARGS)

    assert scm.model is fitted_model
    assert scm.idata is fitted_idata
    xr.testing.assert_identical(scm.idata["posterior"].to_dataset(), posterior_before)


def test_prediction_requires_requested_variables_to_exist(
    conditioning: tuple[SCM, xr.Dataset],
) -> None:
    scm, future = conditioning
    with pytest.raises(ValueError, match="prediction variable"):
        scm.sample_posterior_predictive(
            future, var_names="misspelled_output", **PREDICT_KWARGS
        )


def test_prediction_cannot_resample_a_missing_fitted_parameter(
    shared_slope: SCM,
) -> None:
    scm = copy(shared_slope)
    posterior_before = shared_slope.idata["posterior"].to_dataset().copy(deep=True)
    scm.idata = shared_slope.idata.copy(deep=True)
    scm.idata["posterior"] = posterior_before.drop_vars("slope")

    with pytest.raises(ValueError, match="slope"):
        scm.sample_posterior_predictive(_shared_slope_training(), **PREDICT_KWARGS)

    xr.testing.assert_identical(
        shared_slope.idata["posterior"].to_dataset(), posterior_before
    )


def test_scm_rejects_invalid_equation_sets():
    with pytest.raises(TypeError):
        SCM()
    with pytest.raises(ValueError, match="name its observations"):
        SCM(Equation(name="latent", mu=Parameter("a", Prior("Normal"))))
    with pytest.raises(ValueError, match="distinct"):
        SCM(
            Equation(observed="y", mu=Parameter("a", Prior("Normal"))),
            Equation(observed="y", mu=Parameter("b", Prior("Normal"))),
        )


def _scalar_training():
    return xr.Dataset(
        {"y": ("date", np.array([1.0, 1.4, 0.8, 1.1]))},
        coords={"date": pd.date_range("2025-01-01", periods=4, freq="D")},
    )


def test_equation_name_defaults_to_observed_variable():
    model = SCM(
        Equation(observed="y", mu=Parameter("level", Prior("Normal")))
    ).build_model(_scalar_training())

    assert [rv.name for rv in model.observed_RVs] == ["y"]


def test_nanosecond_dates_build_with_dot_terms():
    dates = pd.date_range("2025-01-01", periods=4, freq="D").as_unit("ns")
    train = xr.Dataset(
        {"x": (("date", "feature"), np.ones((4, 1))), "y": ("date", np.ones(4))},
        coords={"date": dates, "feature": ["a"]},
    )
    model = SCM(
        Equation(
            observed="y", mu=Dot(var_name="x", prior=Prior("Normal", dims="feature"))
        )
    ).build_model(train)

    assert model.named_vars_to_dims["x_beta"] == ("feature",)


def test_specification_change_after_fit_blocks_prediction():
    likelihood = Prior("Normal", sigma=1)
    scm = SCM(
        Equation(
            observed="y", mu=Parameter("level", Prior("Normal")), likelihood=likelihood
        )
    )
    train = _scalar_training()
    scm.fit(train, **SAMPLE_KWARGS)

    likelihood.parameters["sigma"] = 3

    with pytest.raises(RuntimeError, match="specification changed"):
        scm.sample_posterior_predictive(train, **PREDICT_KWARGS)


def test_failed_rebuild_clears_previous_fit():
    scm = SCM(Equation(observed="y", mu=Parameter("level", Prior("Normal"))))
    train = _scalar_training()
    scm.fit(train, **SAMPLE_KWARGS)

    with pytest.raises(ValueError, match="missing observations 'y'"):
        scm.build_model(train.rename(y="z"))

    assert scm.model is None
    assert scm.idata is None
    with pytest.raises(RuntimeError, match="fit before"):
        scm.sample_posterior_predictive(train, **PREDICT_KWARGS)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"model": None}, id="external-model"),
        pytest.param({"return_inferencedata": False}, id="non-inferencedata-result"),
    ],
)
def test_fit_rejects_arguments_that_bypass_its_model_or_posterior(
    kwargs: dict[str, Any],
) -> None:
    data = _scalar_training()
    scm = SCM(Equation(observed="y", mu=Parameter("level", Prior("Normal"))))

    with pytest.raises(ValueError, match="return_inferencedata"):
        scm.fit(data, **kwargs)

    assert scm.idata is None
    with pytest.raises(RuntimeError, match="fit before"):
        scm.sample_posterior_predictive(data, **PREDICT_KWARGS)
