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
"""Numerical and ownership contracts for experimental media and seasonality terms."""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytest
import xarray as xr
from numpy.testing import assert_allclose
from pymc_extras.prior import Prior

from pymc_marketing.mmm import (
    GeometricAdstock,
    LogisticSaturation,
    NoAdstock,
    WeeklyFourier,
    YearlyFourier,
)
from pymc_marketing.mmm.experimental import Data, MediaTransform, Seasonality
from pymc_marketing.mmm.experimental._graph import BuildContext
from pymc_marketing.mmm.transformers import ConvMode

DATES = pd.date_range("2025-01-06", periods=5, freq="W-MON")
CHANNELS = ["radio", "tv"]
SPEND = np.array([[3.0, 0.5], [1.0, 2.0], [0.0, 1.5], [2.5, 0.0], [0.75, 4.0]])


@pytest.fixture
def ds() -> xr.Dataset:
    return xr.Dataset(
        {"spend": (("date", "channel"), SPEND)},
        coords={"date": DATES, "channel": CHANNELS},
    )


def _geometric_adstock(x: np.ndarray, alpha: object, l_max: int) -> np.ndarray:
    lags = np.arange(l_max).reshape((-1,) + (1,) * np.ndim(alpha))
    weights = np.asarray(alpha, dtype=float) ** lags
    weights = weights / weights.sum(axis=0)
    out = np.zeros_like(x, dtype=float)
    for lag in range(l_max):
        out[lag:] += weights[lag] * x[: len(x) - lag]
    return out


def _logistic_saturation(x: np.ndarray, lam: object, beta: object) -> np.ndarray:
    return beta * (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))


def _fourier_modes(
    dates: pd.DatetimeIndex, n_order: int, days_in_period: float
) -> np.ndarray:
    angle = (
        2 * np.pi * np.outer(dates.dayofyear / days_in_period, range(1, n_order + 1))
    )
    return np.concatenate([np.sin(angle), np.cos(angle)], axis=1)


def _ordered(values: np.ndarray, source: tuple[str, ...], *dims: str) -> np.ndarray:
    return xr.DataArray(values, dims=source).transpose(*dims).values


def test_adstock_then_saturation_matches_numpy_reference(ds):
    term = MediaTransform(
        Data("spend"),
        GeometricAdstock(l_max=3, priors={"alpha": 0.5}),
        LogisticSaturation(priors={"lam": 2.0, "beta": 1.5}),
    )
    with pm.Model() as model:
        value = BuildContext(ds).build(term)
    expected = _logistic_saturation(_geometric_adstock(SPEND, 0.5, 3), 2.0, 1.5)
    assert set(value.dims) == {"date", "channel"}
    assert model.coords["channel"] == ("radio", "tv")
    assert model.free_RVs == []
    assert_allclose(
        _ordered(value.eval(), value.dims, "date", "channel"),
        expected,
        rtol=0,
        atol=1e-12,
    )


def test_saturation_then_adstock_applies_transforms_in_order(ds):
    term = MediaTransform(
        Data("spend"),
        LogisticSaturation(priors={"lam": 2.0, "beta": 1.5}),
        GeometricAdstock(l_max=3, priors={"alpha": 0.5}),
    )
    with pm.Model():
        value = BuildContext(ds).build(term)
    expected = _geometric_adstock(_logistic_saturation(SPEND, 2.0, 1.5), 0.5, 3)
    assert_allclose(
        _ordered(value.eval(), value.dims, "date", "channel"),
        expected,
        rtol=0,
        atol=1e-12,
    )


def test_channel_prior_is_applied_per_channel(ds):
    term = MediaTransform(
        Data("spend"),
        LogisticSaturation(
            priors={"lam": 2.0, "beta": Prior("HalfNormal", sigma=1, dims="channel")}
        ),
    )
    with pm.Model() as model:
        value = BuildContext(ds).build(term)
        evaluate = pytensor.function([model["saturation_beta"]], value)
    assert model.named_vars_to_dims["saturation_beta"] == ("channel",)
    assert set(value.dims) == {"date", "channel"}
    beta = np.array([1.5, 0.25])
    assert_allclose(
        _ordered(evaluate(beta), value.dims, "date", "channel"),
        _logistic_saturation(SPEND, 2.0, beta),
        rtol=0,
        atol=1e-12,
    )


def test_required_history_sums_causal_adstock_stages():
    two_stage = MediaTransform(
        Data("spend"),
        GeometricAdstock(l_max=3),
        GeometricAdstock(l_max=2, prefix="carryover"),
    )
    assert two_stage.required_history == 3
    assert MediaTransform(Data("spend"), NoAdstock(l_max=4)).required_history == 0
    assert MediaTransform(Data("spend"), LogisticSaturation()).required_history == 0


def test_noncausal_adstock_rejects_required_history():
    term = MediaTransform(
        Data("spend"), GeometricAdstock(l_max=3, mode=ConvMode.Before)
    )
    with pytest.raises(ValueError, match="causal"):
        term.required_history


def test_repeated_transformation_kind_needs_distinct_prefixes(ds):
    with pytest.raises(ValueError, match="prefix"):
        MediaTransform(
            Data("spend"), GeometricAdstock(l_max=3), GeometricAdstock(l_max=2)
        )
    term = MediaTransform(
        Data("spend"),
        GeometricAdstock(l_max=3, priors={"alpha": 0.5}),
        GeometricAdstock(l_max=2, prefix="carryover", priors={"alpha": 0.2}),
    )
    with pm.Model():
        value = BuildContext(ds).build(term)
    expected = _geometric_adstock(_geometric_adstock(SPEND, 0.5, 3), 0.2, 2)
    assert_allclose(
        _ordered(value.eval(), value.dims, "date", "channel"),
        expected,
        rtol=0,
        atol=1e-12,
    )


def test_dataarray_constant_is_aligned_by_label_not_position(ds):
    alpha = xr.DataArray(
        [0.2, 0.8], dims="channel", coords={"channel": ["tv", "radio"]}
    )
    adstock = GeometricAdstock(l_max=3)
    adstock.update_priors({"adstock_alpha": alpha})
    with pm.Model() as model:
        value = BuildContext(ds).build(MediaTransform(Data("spend"), adstock))
    assert model.free_RVs == []
    assert_allclose(
        _ordered(value.eval(), value.dims, "date", "channel"),
        _geometric_adstock(SPEND, np.array([0.8, 0.2]), 3),
        rtol=0,
        atol=1e-12,
    )


def test_unlabeled_vector_constant_is_rejected(ds):
    term = MediaTransform(
        Data("spend"), GeometricAdstock(l_max=3, priors={"alpha": [0.2, 0.8]})
    )
    with pm.Model(), pytest.raises(ValueError, match="DataArray"):
        BuildContext(ds).build(term)


def test_prior_dimension_absent_from_dataset_is_rejected(ds):
    term = MediaTransform(
        Data("spend"),
        LogisticSaturation(priors={"beta": Prior("HalfNormal", sigma=1, dims="geo")}),
    )
    with pm.Model(), pytest.raises(ValueError, match="coordinates"):
        BuildContext(ds).build(term)


def test_shared_transformation_instance_across_terms_is_rejected(ds):
    shared = GeometricAdstock(l_max=2)
    left = MediaTransform(Data("spend"), shared)
    right = MediaTransform(Data("spend"), shared)
    with pm.Model(), pytest.raises(ValueError, match="adstock_alpha"):
        context = BuildContext(ds)
        context.build(left)
        context.build(right)


def test_one_term_builds_once_per_context(ds):
    term = MediaTransform(Data("spend"), GeometricAdstock(l_max=2))
    with pm.Model() as model:
        context = BuildContext(ds)
        first = context.build(term)
        second = context.build(term)
    assert first is second
    assert [variable.name for variable in model.free_RVs] == ["adstock_alpha"]


def test_media_without_date_dimension_is_rejected(ds):
    ds = ds.assign(scale=("channel", [1.0, 2.0]))
    term = MediaTransform(Data("scale"), GeometricAdstock(l_max=2))
    with pm.Model(), pytest.raises(ValueError, match="date"):
        BuildContext(ds).build(term)


@pytest.mark.parametrize("transforms", [(), (object(),)])
def test_media_transform_requires_transformations(transforms):
    with pytest.raises(TypeError):
        MediaTransform(Data("spend"), *transforms)


def test_yearly_seasonality_matches_fourier_reference():
    dates = pd.date_range("2025-01-06", periods=6, freq="W-MON")
    ds = xr.Dataset(coords={"date": dates})
    term = Seasonality(YearlyFourier(n_order=2, prior=Prior("Laplace", mu=0, b=1)))
    with pm.Model() as model:
        value = BuildContext(ds).build(term)
        evaluate = pytensor.function([model["fourier_beta"]], value)
    assert model.named_vars_to_dims["fourier_beta"] == ("fourier",)
    assert value.dims == ("date",)
    beta = np.array([0.5, -1.0, 2.0, 0.25])
    assert_allclose(
        evaluate(beta),
        _fourier_modes(dates, 2, 365.25) @ beta,
        rtol=0,
        atol=1e-10,
    )


def test_seasonality_prior_dims_broadcast_over_geo():
    dates = pd.date_range("2025-01-06", periods=6, freq="W-MON")
    ds = xr.Dataset(coords={"date": dates, "geo": ["east", "west"]})
    fourier = YearlyFourier(
        n_order=2, prior=Prior("Laplace", mu=0, b=1, dims=("geo", "fourier"))
    )
    with pm.Model() as model:
        value = BuildContext(ds).build(Seasonality(fourier))
        evaluate = pytensor.function([model["fourier_beta"]], value)
    assert model.named_vars_to_dims["fourier_beta"] == ("geo", "fourier")
    assert model.coords["geo"] == ("east", "west")
    assert set(value.dims) == {"date", "geo"}
    beta = np.array([[0.5, -1.0, 2.0, 0.25], [1.0, 0.0, -0.5, 3.0]])
    assert_allclose(
        _ordered(evaluate(beta), value.dims, "date", "geo"),
        _fourier_modes(dates, 2, 365.25) @ beta.T,
        rtol=0,
        atol=1e-10,
    )


def test_seasonality_requires_date_dimension():
    ds = xr.Dataset(coords={"channel": ["radio"]})
    with pm.Model(), pytest.raises(ValueError, match="date"):
        BuildContext(ds).build(Seasonality(YearlyFourier(n_order=1)))


def test_seasonality_requires_fourier_component():
    with pytest.raises(TypeError):
        Seasonality(object())


def test_seasonality_prefixes_must_be_distinct_within_a_context():
    dates = pd.date_range("2025-01-06", periods=6, freq="W-MON")
    ds = xr.Dataset(coords={"date": dates})
    with pm.Model(), pytest.raises(ValueError, match="fourier_beta"):
        context = BuildContext(ds)
        context.build(Seasonality(YearlyFourier(n_order=2)))
        context.build(Seasonality(YearlyFourier(n_order=1)))
    with pm.Model() as model:
        context = BuildContext(ds)
        yearly = context.build(Seasonality(YearlyFourier(n_order=2)))
        weekly = context.build(Seasonality(WeeklyFourier(n_order=1, prefix="weekly")))
        evaluate = pytensor.function(
            [model["fourier_beta"], model["weekly_beta"]], yearly + weekly
        )
    assert model.named_vars_to_dims["weekly_beta"] == ("weekly",)
    yearly_beta = np.array([0.5, -1.0, 2.0, 0.25])
    weekly_beta = np.array([3.0, -2.0])
    assert_allclose(
        evaluate(yearly_beta, weekly_beta),
        _fourier_modes(dates, 2, 365.25) @ yearly_beta
        + _fourier_modes(dates, 1, 7.0) @ weekly_beta,
        rtol=0,
        atol=1e-10,
    )
