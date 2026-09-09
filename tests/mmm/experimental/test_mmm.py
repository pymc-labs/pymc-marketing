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
"""Behavioral contracts for experimental construction and posterior prediction."""

import numpy as np
import pandas as pd
import pymc.dims as pmd
import pytest
import xarray as xr
from pymc_extras.prior import Prior

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.experimental import MMM, Data, Equation
from pymc_marketing.mmm.scaling import FixedScaling
from pymc_marketing.terms import Dot, Parameter, Transform

SAMPLE_KWARGS = {
    "draws": 30,
    "tune": 40,
    "chains": 1,
    "cores": 1,
    "random_seed": 892,
    "progressbar": False,
    "compute_convergence_checks": False,
}


def _lag(values):
    return pmd.concat(
        [
            pmd.zeros_like(values.isel(date=slice(0, 1))),
            values.isel(date=slice(None, -1)),
        ],
        dim="date",
    )


class Lag(Transform):
    required_history = 1


def test_complete_outcome_replacement_and_shared_parameter():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=3),
            "price": [1.0, 2.0, 3.0],
            "orders": [5.0, 6.0, 7.0],
        }
    )
    shared = Parameter("shared_level", Prior("Normal"))
    mmm = MMM(
        date_column="timestamp",
        target_column="unused_target",
        channel_columns=["unused_channel"],
        yearly_seasonality=3,
    )
    mmm.y = Equation(
        name="purchase_process",
        observed="orders",
        mu=shared + shared + Data("price"),
        likelihood=Prior("Normal", sigma=1),
    )
    model = mmm.build_model(data)
    # Replacement must not retain any of the default stochastic components.
    assert {rv.name for rv in model.free_RVs} == {"shared_level"}
    assert {rv.name for rv in model.observed_RVs} == {"purchase_process"}
    expected = -len(data) * np.log(2 * np.pi) / 2
    actual = model.compile_logp(vars=model.observed_RVs)({"shared_level": 2.0})
    np.testing.assert_allclose(actual, expected)


def test_joint_fit_custom_history_and_conditioned_mechanism_pruning():
    tv = np.linspace(0.5, 2.0, 8)
    train = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=8),
            "tv": tv,
            "search": 2 * np.r_[0, tv[:-1]],
            "revenue": 6 * np.r_[0, tv[:-1]],
        }
    )
    mmm = MMM(channel_columns=["tv", "search"])
    mmm.media["search"].equation = Equation(
        name="demand_process",
        mu=2 * Lag(mmm.media["tv"].value, func=_lag),
        likelihood=Prior("Normal", sigma=0.001),
    )
    mmm.y = Equation(
        name="purchase_process",
        observed="revenue",
        mu=Parameter("response", Prior("Normal", mu=3, sigma=0.001))
        * mmm.media["search"].value,
        likelihood=Prior("Normal", sigma=0.01),
    )
    idata = mmm.fit(train, **SAMPLE_KWARGS)
    observations = idata["observed_data"].dataset.copy(deep=True)
    assert set(observations) == {"demand_process", "purchase_process"}
    future = pd.DataFrame(
        {"date": pd.date_range("2026-01-09", periods=2), "tv": [3.0, 4.0]}
    )
    generated = mmm.sample_posterior_predictive(
        future, condition_on=(), random_seed=34, progressbar=False
    )
    expected_demand = xr.DataArray(
        [4.0, 6.0], dims="date", coords={"date": future.date}
    )
    np.testing.assert_allclose(
        generated["demand_process"],
        expected_demand.broadcast_like(generated["demand_process"]),
        atol=0.01,
    )
    np.testing.assert_allclose(
        generated["purchase_process"], 3 * generated["demand_process"], atol=0.1
    )
    # Conditioning discards the lag mechanism: no TV input or contiguous dates required.
    supplied = pd.DataFrame(
        {"date": pd.date_range("2027-01-01", periods=2), "search": [40.0, 50.0]}
    )
    conditioned = mmm.sample_posterior_predictive(
        supplied, condition_on=["search"], random_seed=35, progressbar=False
    )
    expected_fixed = xr.DataArray(
        [40.0, 50.0], dims="date", coords={"date": supplied.date}
    )
    xr.testing.assert_allclose(
        conditioned["demand_process"],
        expected_fixed.broadcast_like(conditioned["demand_process"]),
    )
    np.testing.assert_allclose(
        conditioned["purchase_process"], 3 * conditioned["demand_process"], atol=0.3
    )
    xr.testing.assert_identical(idata["observed_data"].to_dataset(), observations)
    with pytest.raises(ValueError):
        mmm.sample_posterior_predictive(future, progressbar=False)
    # In-place prior edits cannot quietly reinterpret the fitted posterior.
    mmm.y.likelihood.parameters["sigma"] = 2.0
    with pytest.raises(RuntimeError, match="specification changed"):
        mmm.sample_posterior_predictive(future, condition_on=(), progressbar=False)


def test_refit_uses_current_data_and_failed_refit_invalidates():
    data = pd.DataFrame(
        {"date": pd.date_range("2026-01-01", periods=8), "count": np.full(8, 2.0)}
    )
    mmm = MMM()
    mmm.y = Equation(
        name="count_process",
        observed="count",
        mu=Parameter("level", Prior("Normal", sigma=10)),
        likelihood=Prior("Normal", sigma=0.02),
    )
    first = mmm.fit(data, **SAMPLE_KWARGS)["posterior"].dataset["level"].mean().item()
    second = (
        mmm.fit(data.assign(count=7.0), **SAMPLE_KWARGS)["posterior"]
        .dataset["level"]
        .mean()
        .item()
    )
    assert second > first + 4.0
    with pytest.raises((KeyError, ValueError)):
        mmm.fit(data.drop(columns="count"), **SAMPLE_KWARGS)
    with pytest.raises(RuntimeError, match="fit"):
        mmm.sample_posterior_predictive(data, progressbar=False)


def test_conflicting_observation_bindings_are_not_silently_overwritten():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=3),
            "a": [1.0] * 3,
            "b": [2.0] * 3,
            "y": [3.0] * 3,
        }
    )
    mmm = MMM(channel_columns=["a", "b"])
    equation = Equation(mu=0, likelihood=Prior("Normal", sigma=1))
    mmm.media["a"].equation = equation
    mmm.media["b"].equation = equation
    with pytest.raises(ValueError, match="different observation"):
        mmm.build_model(data)


def test_no_history_prediction_returns_original_units_and_rejects_changed_scales():
    train = pd.DataFrame(
        {"date": pd.date_range("2026-01-01", periods=8), "y": np.full(8, 2.0)}
    )
    mmm = MMM(
        scaling={"target": FixedScaling(value=10.0, dims=())},
        model_config={"likelihood": Prior("Normal", sigma=0.001)},
    )
    idata = mmm.fit(train, **SAMPLE_KWARGS)
    predicted = mmm.sample_posterior_predictive(random_seed=19, progressbar=False)
    expected = (10 * idata["posterior"].dataset["intercept"]).broadcast_like(
        predicted.y
    )
    xr.testing.assert_allclose(predicted.y, expected, atol=0.05)
    mmm.scalers["target_scale"] = 2 * mmm.scalers["target_scale"]
    with pytest.raises(RuntimeError):
        mmm.sample_posterior_predictive(progressbar=False)


def test_feature_labels_preserve_posterior_coefficient_meaning():
    features = np.tile(np.eye(2), (4, 1))
    train = xr.Dataset(
        {
            "controls": (("date", "feature"), features),
            "orders": ("date", features @ np.array([2.0, -1.0])),
        },
        coords={"date": pd.date_range("2026-01-01", periods=8), "feature": ["a", "b"]},
    )
    mmm = MMM()
    mmm.y = Equation(
        name="purchases",
        observed="orders",
        mu=Dot(var_name="controls", prior=Prior("Normal", dims="feature")),
        likelihood=Prior("Normal", sigma=0.001),
    )
    idata = mmm.fit(train, **SAMPLE_KWARGS)
    future = xr.Dataset(
        {"controls": (("date", "feature"), [[3.0, 1.0], [2.0, 5.0]])},
        coords={"date": pd.date_range("2026-01-09", periods=2), "feature": ["a", "b"]},
    )
    predicted = mmm.sample_posterior_predictive(
        future, random_seed=38, progressbar=False
    )
    reordered = mmm.sample_posterior_predictive(
        future.sel(feature=["b", "a"]), random_seed=38, progressbar=False
    )
    xr.testing.assert_equal(predicted, reordered)
    expected = xr.dot(
        future.controls, idata["posterior"].dataset["controls_beta"], dim="feature"
    ).transpose(*predicted.purchases.dims)
    xr.testing.assert_allclose(predicted.purchases, expected, atol=0.01)


@pytest.mark.parametrize("reverse", [False, True])
def test_media_forecast_uses_training_scales_and_measured_history(reverse):
    def reference(values):
        if reverse:
            values = 0.8 * np.tanh(0.6 * values / 2)
        carried = values + 0.5 * np.r_[0, values[:-1]]
        return carried if reverse else 0.8 * np.tanh(0.6 * carried / 2)

    spend = np.arange(1, 9, dtype=float)
    train = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=8),
            "tv": spend,
            "sales": 10 * (0.3 + reference(spend / spend.max())),
        }
    )
    transforms = (
        GeometricAdstock(l_max=2, normalize=False, priors={"alpha": 0.5}),
        LogisticSaturation(priors={"lam": 0.6, "beta": 0.8}),
    )
    mmm = MMM(
        target_column="sales",
        channel_columns=["tv"],
        media_transform=transforms[::-1] if reverse else transforms,
        scaling={"target": FixedScaling(value=10.0, dims=())},
        model_config={"likelihood": Prior("Normal", sigma=0.0001)},
    )
    idata = mmm.fit(train, **SAMPLE_KWARGS)
    future = pd.DataFrame(
        {"date": pd.date_range("2026-01-09", periods=2), "tv": [16.0, 18.0]}
    )
    predicted = mmm.sample_posterior_predictive(
        future, random_seed=40, progressbar=False
    )
    contribution = xr.DataArray(
        reference(np.r_[spend[-1], future.tv] / spend.max())[1:],
        dims="date",
        coords={"date": future.date},
    )
    expected = 10 * (idata["posterior"].dataset["intercept"] + contribution)
    xr.testing.assert_allclose(predicted.sales, expected, atol=0.01)


def test_generated_intermediate_retains_inferred_training_dimensions():
    drivers = np.arange(12, dtype=float).reshape(6, 2)
    train = xr.Dataset(
        {
            "drivers": (("date", "feature"), drivers),
            "controls": (("date", "feature"), drivers + 0.1),
            "orders": ("date", (drivers + 0.1).sum(axis=-1)),
        },
        coords={"date": pd.date_range("2026-01-01", periods=6), "feature": ["a", "b"]},
    )
    controls = Equation(
        name="control_process",
        observed="controls",
        mu=Data("drivers"),
        likelihood=Prior("Normal", sigma=0.001),
    )
    mmm = MMM()
    mmm.y = Equation(
        name="purchases",
        observed="orders",
        mu=Transform(controls, func=lambda value: value.sum("feature"))
        + Parameter("level", Prior("Normal", sigma=0.001)),
        likelihood=Prior("Normal", sigma=0.001),
    )
    mmm.fit(train, **SAMPLE_KWARGS)
    future = xr.Dataset(
        {"drivers": (("date", "feature"), [[20.0, 30.0], [40.0, 50.0]])},
        coords={"date": pd.date_range("2026-01-07", periods=2), "feature": ["a", "b"]},
    )
    predicted = mmm.sample_posterior_predictive(
        future, condition_on=(), random_seed=18, progressbar=False
    )
    xr.testing.assert_allclose(
        predicted.control_process,
        future.drivers.broadcast_like(predicted.control_process),
        atol=0.01,
    )
    xr.testing.assert_allclose(
        predicted.purchases, predicted.control_process.sum("feature"), atol=0.01
    )


def test_prediction_keeps_fitted_coordinates_not_present_in_future_inputs():
    train = xr.Dataset(
        {"orders": ("date", np.full(8, 3.0))},
        coords={"date": pd.date_range("2026-01-01", periods=8), "feature": ["a", "b"]},
    )
    mmm = MMM()
    mmm.y = Equation(
        name="purchases",
        observed="orders",
        mu=Transform(
            Parameter("levels", Prior("Normal", dims="feature")),
            func=lambda value: value.sum("feature"),
        ),
        likelihood=Prior("Normal", sigma=0.001),
    )
    idata = mmm.fit(train, **SAMPLE_KWARGS)
    future = pd.DataFrame({"date": pd.date_range("2026-01-09", periods=2)})
    predicted = mmm.sample_posterior_predictive(
        future, var_names=["purchases", "levels"], random_seed=21, progressbar=False
    )
    xr.testing.assert_equal(predicted.levels, idata["posterior"].dataset["levels"])
    expected = predicted.levels.sum("feature").broadcast_like(predicted.purchases)
    xr.testing.assert_allclose(predicted.purchases, expected, atol=0.01)


def test_date_parameters_are_reused_in_sample_but_require_a_forecast_mechanism():
    train = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=6),
            "orders": np.arange(1, 7, dtype=float),
        }
    )
    mmm = MMM()
    mmm.y = Equation(
        name="purchases",
        observed="orders",
        mu=Parameter("daily_level", Prior("Normal", sigma=10, dims="date")),
        likelihood=Prior("Normal", sigma=0.001),
    )
    idata = mmm.fit(train, **SAMPLE_KWARGS)
    predicted = mmm.sample_posterior_predictive(random_seed=59, progressbar=False)
    xr.testing.assert_allclose(
        predicted.purchases, idata["posterior"].dataset["daily_level"], atol=0.01
    )
    subset = train.iloc[[1, 4]][["date"]]
    selected = mmm.sample_posterior_predictive(
        subset, random_seed=60, progressbar=False
    )
    xr.testing.assert_allclose(
        selected.purchases,
        idata["posterior"].dataset["daily_level"].sel(date=subset.date.to_numpy()),
        atol=0.01,
    )
    future = pd.DataFrame({"date": pd.date_range("2026-01-07", periods=2)})
    with pytest.raises(ValueError):
        mmm.sample_posterior_predictive(future, progressbar=False)
