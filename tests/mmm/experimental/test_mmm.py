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
"""Behavioral contracts of the experimental graph-first MMM: fitting and forecasting."""

import numpy as np
import pandas as pd
import pymc.dims as pmd
import pytest
import xarray as xr
from pymc_extras.prior import Prior
from scipy.special import gammaln
from scipy.stats import norm

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.experimental import MMM, Data, Equation, MediaTransform
from pymc_marketing.terms import Intercept, Parameter, Transform

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
    response = MediaTransform(
        Data("spend"),
        GeometricAdstock(l_max=L_MAX, priors={"alpha": ALPHA}),
        LogisticSaturation(
            priors={
                "lam": LAM,
                "beta": Prior("HalfNormal", sigma=1, dims=("channel", "target")),
            }
        ),
    )
    mean = (
        Intercept(prior=Prior("Normal", dims=("product", "target")))
        + Transform(response, lambda value: value.sum(dim="channel"))
        + Data("price") * Parameter("price_beta", Prior("Normal", dims="target"))
    )
    return Equation(
        observed="sales",
        mu=Transform(
            mean,
            lambda value: pmd.Deterministic(
                "sales_mean", value, dims=("date", "product", "target")
            ),
        ),
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
    mmm = MMM(_multi_target_recipe())
    mmm.fit(train, **SAMPLE_KWARGS)
    return mmm, train, future


def _forecast_mean(mmm, future, **kwargs):
    return mmm.sample_posterior_predictive(
        future, var_names=["sales", "sales_mean"], **PREDICT_KWARGS, **kwargs
    )


def _oracle_with_history(mmm, train, future):
    spend = np.concatenate(
        [train["spend"].values[-(L_MAX - 1) :], future["spend"].values]
    )
    return _sales_mean_oracle(
        mmm.idata["posterior"].to_dataset(), spend, future["price"].values, L_MAX - 1
    )


def test_multi_target_forecast_matches_numpy_oracle(multi_target):
    mmm, train, future = multi_target
    prediction = _forecast_mean(mmm, future)

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
        _oracle_with_history(mmm, train, future),
        atol=1e-10,
    )


def test_forecast_is_invariant_to_label_order(multi_target):
    mmm, _, future = multi_target
    reversed_labels = _multi_target_future(
        future["spend"].values[:, ::-1],
        future["price"].values,
        channel=CHANNELS[::-1],
        coords={"target": TARGETS[::-1]},
    )

    baseline = _forecast_mean(mmm, future)["sales_mean"]
    reordered = _forecast_mean(mmm, reversed_labels)["sales_mean"]

    assert list(reordered["target"].values) == TARGETS
    np.testing.assert_allclose(reordered.values, baseline.values, atol=1e-10)


def test_forecast_responds_to_future_spend(multi_target):
    mmm, train, future = multi_target
    scaled = future.assign(spend=future["spend"] * 1.7)

    baseline = _forecast_mean(mmm, future)["sales_mean"]
    prediction = _forecast_mean(mmm, scaled)["sales_mean"]

    assert not np.allclose(prediction.values, baseline.values)
    np.testing.assert_allclose(
        prediction.transpose("chain", "draw", "date", "product", "target").values,
        _oracle_with_history(mmm, train, scaled),
        atol=1e-10,
    )


def test_forecast_without_history_pads_with_zeros(multi_target):
    mmm, _, future = multi_target

    with_history = _forecast_mean(mmm, future)["sales_mean"]
    scenario = _forecast_mean(mmm, future, include_last_observations=False)[
        "sales_mean"
    ]

    assert not np.allclose(scenario.values, with_history.values)
    np.testing.assert_allclose(
        scenario.transpose("chain", "draw", "date", "product", "target").values,
        _sales_mean_oracle(
            mmm.idata["posterior"].to_dataset(),
            future["spend"].values,
            future["price"].values,
            0,
        ),
        atol=1e-10,
    )


@pytest.mark.parametrize(
    ("alter", "match"),
    [
        pytest.param(
            lambda future: future.assign_coords(
                date=future["date"] + pd.Timedelta(weeks=1)
            ),
            "immediately follow training",
            id="shifted-dates",
        ),
        pytest.param(
            lambda future: future.assign_coords(target=TARGETS[:1]),
            "labels for 'target' must match training",
            id="target-subset",
        ),
        pytest.param(
            lambda future: future.drop_vars("channel"),
            "'channel' must provide coordinate labels",
            id="unlabeled-channel",
        ),
    ],
)
def test_forecast_rejects_inconsistent_future_data(multi_target, alter, match):
    mmm, _, future = multi_target
    with pytest.raises(ValueError, match=match):
        _forecast_mean(mmm, alter(future))


def test_build_model_requires_every_observation_variable():
    train = _multi_target_training().drop_vars("sales")
    with pytest.raises(ValueError, match="missing observations 'sales'"):
        MMM(_multi_target_recipe()).build_model(train)


def test_forecast_leaves_fit_untouched(multi_target):
    mmm, train, future = multi_target
    fitted_model = mmm.model
    train_before = train.copy(deep=True)
    posterior_before = mmm.idata["posterior"].to_dataset().copy(deep=True)

    _forecast_mean(mmm, future.isel(channel=slice(None, None, -1)))

    assert mmm.model is fitted_model
    xr.testing.assert_identical(train, train_before)
    xr.testing.assert_identical(mmm.idata["posterior"].to_dataset(), posterior_before)


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
        mu=Transform(
            slope * Data("driver"),
            lambda value: pmd.Deterministic("revenue_mean", 2.0 + value),
        ),
        likelihood=Prior("Normal", sigma=0.1),
    )
    orders = Equation(
        name="orders",
        observed="orders_obs",
        mu=Transform(
            slope * Data("driver"),
            lambda value: pmd.Deterministic(
                "order_rate", pmd.math.exp(0.3 + 0.2 * value)
            ),
        ),
        likelihood=Prior("Poisson"),
    )
    calibration = Equation(
        name="calibration",
        observed="lift",
        mu=slope,
        likelihood=Prior("Normal", sigma=0.2),
    )
    mmm = MMM(revenue, orders, calibration)
    mmm.fit(_shared_slope_training(), **SAMPLE_KWARGS)
    return mmm


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
    mmm = MMM(search, sales)
    mmm.fit(_conditioning_training(), **SAMPLE_KWARGS)
    future = xr.Dataset(
        {
            "tv": ("date", np.array([1.0, 2.0, 3.0])),
            "search": ("date", np.array([2.2, 3.9, 6.1])),
        },
        coords={"date": pd.date_range("2025-06-09", periods=3, freq="D")},
    )
    return mmm, future


def test_condition_on_holds_supplied_observations_fixed(conditioning):
    mmm, future = conditioning

    conditioned = mmm.sample_posterior_predictive(
        future, condition_on=["search"], **PREDICT_KWARGS
    )
    generated = mmm.sample_posterior_predictive(
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
    ],
)
def test_condition_on_rejects_invalid_selectors(conditioning, condition_on, error):
    mmm, future = conditioning
    with pytest.raises(error):
        mmm.sample_posterior_predictive(
            future, condition_on=condition_on, **PREDICT_KWARGS
        )


def test_mmm_rejects_invalid_equation_sets():
    with pytest.raises(TypeError):
        MMM()
    with pytest.raises(ValueError, match="name its observations"):
        MMM(Equation(name="latent", mu=Parameter("a", Prior("Normal"))))
    with pytest.raises(ValueError, match="distinct"):
        MMM(
            Equation(observed="y", mu=Parameter("a", Prior("Normal"))),
            Equation(observed="y", mu=Parameter("b", Prior("Normal"))),
        )


def _scalar_training():
    return xr.Dataset(
        {"y": ("date", np.array([1.0, 1.4, 0.8, 1.1]))},
        coords={"date": pd.date_range("2025-01-01", periods=4, freq="D")},
    )


def test_equation_name_defaults_to_observed_variable():
    model = MMM(
        Equation(observed="y", mu=Parameter("level", Prior("Normal")))
    ).build_model(_scalar_training())

    assert [rv.name for rv in model.observed_RVs] == ["y"]


def test_specification_change_after_fit_blocks_prediction():
    likelihood = Prior("Normal", sigma=1)
    mmm = MMM(
        Equation(
            observed="y", mu=Parameter("level", Prior("Normal")), likelihood=likelihood
        )
    )
    train = _scalar_training()
    mmm.fit(train, **SAMPLE_KWARGS)

    likelihood.parameters["sigma"] = 3

    with pytest.raises(RuntimeError, match="specification changed"):
        mmm.sample_posterior_predictive(train, **PREDICT_KWARGS)


def test_failed_rebuild_clears_previous_fit():
    mmm = MMM(Equation(observed="y", mu=Parameter("level", Prior("Normal"))))
    train = _scalar_training()
    mmm.fit(train, **SAMPLE_KWARGS)

    with pytest.raises(ValueError, match="missing observations 'y'"):
        mmm.build_model(train.rename(y="z"))

    assert mmm.model is None
    assert mmm.idata is None
    with pytest.raises(RuntimeError, match="fit before"):
        mmm.sample_posterior_predictive(train, **PREDICT_KWARGS)
