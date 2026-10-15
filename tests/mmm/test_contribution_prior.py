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
"""Statistical and lifecycle contract for fixed-training contribution factors."""

import copy
import json
import warnings

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from pymc.model.transform.optimization import freeze_dims_and_data
from pymc_extras.prior import Prior
from scipy.stats import norm

from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_marketing.mmm.scaling import DataDerivedScaling, Scaling


def make_model(dims=(), **kwargs):
    return MMM(
        date_column="date",
        channel_columns=["search", "tiktok"],
        adstock=GeometricAdstock(l_max=2),
        saturation=LogisticSaturation(),
        dims=dims,
        **kwargs,
    )


@pytest.fixture
def data():
    X = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=5),
            "search": [1.0, 3.0, 2.0, 4.0, 5.0],
            "tiktok": [2.0, 4.0, 3.0, 5.0, 6.0],
        }
    )
    return X, pd.Series([10.0, 20.0, np.nan, 30.0, 40.0])


def evaluate(model, expressions, point=None):
    expressions = [v.values if hasattr(v, "values") else v for v in expressions]
    fn = model.compile_fn(
        model.replace_rvs_by_values(expressions),
        inputs=model.value_vars,
        on_unused_input="ignore",
    )
    return fn(model.initial_point() if point is None else point)


@pytest.mark.parametrize("panel", [False, True])
def test_aggregation_logdensity_and_unchanged_likelihood(data, panel):
    X, y = data
    if panel:
        X = pd.concat(
            [X.assign(country="A"), X.assign(country="B", search=X.search * 2)],
            ignore_index=True,
        )
        y = pd.concat([y, y * 7], ignore_index=True)
    m = make_model(
        dims=("country",) if panel else (),
        scaling=Scaling(
            target=DataDerivedScaling(method="max", dims=()),
            channel=DataDerivedScaling(method="max", dims=()),
        ),
    )
    m.build_model(X, y)
    point = m.model.initial_point()
    base_logp = m.model.compile_logp()(point)
    likelihood = evaluate(m.model, [m.model.logp(vars=m.model.observed_RVs)], point)[0]
    m.add_original_scale_contribution_variable(["channel_contribution"])
    contributions = evaluate(
        m.model, [m.model["channel_contribution_original_scale"]], point
    )[0]
    mask = m._contribution_observed_mask.values
    expected_channel = np.where(mask[..., None], contributions, 0).sum(
        axis=tuple(range(contributions.ndim - 1))
    )
    denominator = np.nansum(y)
    for name, channels, scale in [
        ("all_share", ["search", "tiktok"], "share"),
        ("individual", ["tiktok"], "total"),
        ("nested", ["tiktok"], "share"),
    ]:
        m.add_contribution_prior(
            name, channels, scale, Prior("Normal", mu=0.3, sigma=2)
        )
    actual = evaluate(
        m.model, [m.model[n] for n in ["all_share", "individual", "nested"]], point
    )
    expected = [
        expected_channel.sum() / denominator,
        expected_channel[1],
        expected_channel[1] / denominator,
    ]
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_allclose(
        m.model.compile_logp()(point) - base_logp, norm.logpdf(expected, 0.3, 2).sum()
    )
    np.testing.assert_allclose(
        evaluate(m.model, [m.model.logp(vars=m.model.observed_RVs)], point)[0],
        likelihood,
    )
    # No extra RV, repeated likelihood, renormalization, or duplicated full array.
    assert len(m.model.potentials) == 3
    assert len(m.model.observed_RVs) == 1
    assert len(m.model.free_RVs) == 5


def test_frozen_reference_prediction_optimization_and_rebuild(data):
    X, y = data
    m = make_model()
    m.build_model(X, y)
    m.add_contribution_prior(
        "belief", ["search"], "share", Prior("Normal", mu=0.3, sigma=0.1)
    )
    before = evaluate(
        m.model, [m.model["belief"], m.model.logp(vars=m.model.potentials)]
    )
    reference = copy.deepcopy(m._contribution_reference)
    # Change both dates and data, as prediction and optimizer windows do.
    with m.model:
        pm.set_data(
            {"channel_data": np.full((8, 2), 1000.0), "target_data": np.full(8, 999.0)},
            coords={"date": pd.date_range("2026-01-01", periods=8)},
        )
    after = evaluate(
        m.model, [m.model["belief"], m.model.logp(vars=m.model.potentials)]
    )
    np.testing.assert_allclose(after, before)
    assert m._contribution_reference == reference
    # Adding another component after prediction must still use training inputs.
    m.add_contribution_prior(
        "second", ["search"], "share", Prior("Normal", mu=0.3, sigma=0.1)
    )
    np.testing.assert_allclose(evaluate(m.model, [m.model["second"]])[0], before[0])
    m.build_model(X, y)
    np.testing.assert_allclose(evaluate(m.model, [m.model["belief"]])[0], before[0])
    assert len(m.model.potentials) == 2
    original_model = m.model
    with pytest.raises(ValueError, match="original training"):
        m.build_model(X.assign(search=X.search * 2), y)
    assert m.model is original_model
    with pytest.raises(ValueError, match="original training"):
        m.fit(X.assign(search=X.search * 2), y)
    assert m.model is original_model


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"name": "bad name"}, "identifier"),
        ({"name": "mu"}, "already in use"),
        ({"name": "channel_contribution_original_scale"}, "already in use"),
        ({"channels": []}, "nonempty"),
        ({"channels": "search"}, "nonempty"),
        ({"channels": ["search", "search"]}, "distinct"),
        ({"channels": ["missing"]}, "unknown"),
        ({"scale": "average"}, "scale"),
        ({"prior": Prior("Normal", mu=[0.2, 0.3], sigma=1)}, "fixed finite scalars"),
        (
            {"prior": Prior("Normal", mu=Prior("Normal"), sigma=1)},
            "fixed finite scalars",
        ),
        ({"prior": Prior("Normal", dims="channel")}, "scalar Prior"),
        ({"prior": Prior("Normal", transform="exp")}, "scalar Prior"),
        ({"prior": Prior("Uniform", lower=0, upper=0.05)}, "hard caps"),
        ({"prior": Prior("Beta", alpha=2, beta=5)}, "including Beta"),
        ({"prior": Prior("Normal", mu=0.2, sigma=-1)}, "sigma"),
    ],
)
def test_invalid_input_does_not_mutate_graph(data, kwargs, match):
    m = make_model()
    m.build_model(*data)
    names = set(m.model.named_vars)
    values = {
        "name": "belief",
        "channels": ["search"],
        "scale": "share",
        "prior": Prior("Normal", mu=0.3, sigma=0.1),
    } | kwargs
    with pytest.raises(ValueError, match=match):
        m.add_contribution_prior(**values)
    assert set(m.model.named_vars) == names
    assert m._contribution_priors == []


def test_duplicates_and_unsupported_models(data):
    m = make_model()
    with pytest.raises(ValueError, match="built"):
        m.add_contribution_prior("belief", ["search"], "share", Prior("Normal"))
    m.build_model(*data)
    m.add_contribution_prior("belief", ["search"], "share", Prior("Normal"))
    names = set(m.model.named_vars)
    with pytest.raises(ValueError, match="already in use"):
        m.add_contribution_prior("belief", ["search"], "share", Prior("Normal"))
    assert set(m.model.named_vars) == names
    m.link = "log"
    with pytest.raises(ValueError, match="identity-link"):
        m.add_contribution_prior("other", ["search"], "share", Prior("Normal"))
    m.link = "identity"
    m.mu_effects = [object()]
    with pytest.raises(ValueError, match="MuEffects"):
        m.add_contribution_prior("other", ["search"], "share", Prior("Normal"))


@pytest.mark.parametrize("target", [np.zeros(5), -np.ones(5), np.full(5, np.nan)])
def test_invalid_denominator(data, target):
    m = make_model()
    m.build_model(data[0], pd.Series(target))
    names = set(m.model.named_vars)
    with pytest.raises(ValueError, match="finite positive"):
        m.add_contribution_prior("belief", ["search"], "share", Prior("Normal"))
    assert set(m.model.named_vars) == names


def test_save_load_restore_original_missingness(data, tmp_path):
    X, y = data
    m = make_model()
    m.build_model(X, y)
    m.add_contribution_prior(
        "belief", ["search"], "share", Prior("Normal", mu=0.3, sigma=0.1)
    )
    before = evaluate(
        m.model, [m.model["belief"], m.model.logp(vars=m.model.potentials)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.sample_prior_predictive(X, y, samples=3, random_seed=123)
    # Serialization needs a posterior group; forward draws suffice for this graph round-trip.
    m.idata["posterior"] = m.idata.prior.dataset
    m.idata["fit_data"] = m.create_fit_data(X, y.copy())
    # Simulate the zero-filled fit_data case too: saved mask must win.
    m.idata["fit_data"]["y"] = m.idata.fit_data["y"].fillna(0)
    m.save(tmp_path / "belief.nc")
    loaded = MMM.load(tmp_path / "belief.nc")
    assert loaded._contribution_priors == m._contribution_priors
    assert loaded._contribution_reference == m._contribution_reference
    np.testing.assert_allclose(
        evaluate(
            loaded.model,
            [loaded.model["belief"], loaded.model.logp(vars=loaded.model.potentials)],
        ),
        before,
    )
    loaded.build_from_idata(loaded.idata)
    assert len(loaded.model.potentials) == 1
    assert np.isfinite(loaded.model.compile_dlogp()(loaded.model.initial_point())).all()
    assert (
        json.loads(loaded.idata.attrs["contribution_reference"])["training"]["mask"][2]
        is False
    )


def test_effective_prior_removes_sales_likelihood_retains_factors(data):
    m = make_model()
    m.build_model(*data)
    m.add_contribution_prior(
        "belief", ["search"], "share", Prior("Normal", mu=0.3, sigma=0.1)
    )
    frozen = freeze_dims_and_data(m.model)
    prior_model = pm.do(
        frozen,
        {"y": frozen["y"].type.constant_type(type=frozen["y"].type, data=np.zeros(5))},
        make_interventions_shared=False,
    )
    assert not prior_model.observed_RVs
    assert len(prior_model.potentials) == 1
    point = prior_model.initial_point()
    assert np.isfinite(prior_model.compile_dlogp()(point)).all()
    actual = prior_model.compile_logp()(point)
    parameter_logp = evaluate(
        prior_model, [prior_model.logp(vars=prior_model.free_RVs)], point
    )[0]
    share = evaluate(prior_model, [prior_model["belief"]], point)[0]
    np.testing.assert_allclose(actual - parameter_logp, norm.logpdf(share, 0.3, 0.1))
    with pytest.warns(UserWarning, match="not the effective prior"):
        m.sample_prior_predictive(*data, samples=2, random_seed=123)


@pytest.fixture
def labeled_panel():
    dates = pd.date_range("2025-01-01", periods=5)
    coords = {
        "date": dates,
        "region": ["A", "B"],
        "product": ["p", "q"],
        "channel": ["tiktok", "search"],
    }
    ds = xr.Dataset(
        {
            "_channel": xr.DataArray(
                np.arange(40).reshape(2, 2, 5, 2) + 1.0,
                dims=("channel", "product", "date", "region"),
                coords=coords,
            ),
            "_target": xr.DataArray(
                np.arange(20).reshape(2, 5, 2) + 20.0,
                dims=("region", "date", "product"),
                coords={d: coords[d] for d in ["region", "date", "product"]},
            ),
        }
    )
    ds["_target"].loc[{"region": "B", "product": "q", "date": dates[2]}] = np.nan
    channels = ds["_channel"].to_series().unstack("channel").reset_index()
    targets = ds["_target"].to_series().rename("y").reset_index()
    frame = channels.merge(targets, on=["date", "region", "product"])
    return ds, frame.drop(columns="y"), frame["y"]


def test_xarray_two_dimensions_and_missing_cells(labeled_panel):
    """Label alignment, rather than axis position, defines pooled training cells."""
    ds, _, _ = labeled_panel
    m = make_model(dims=("region", "product"))
    m.build_model(ds)
    m.add_contribution_prior("pooled", ["search"], "share", Prior("Normal"))
    source = m.model["channel_contribution_original_scale"]
    contributions = xr.DataArray(
        evaluate(m.model, [source])[0],
        dims=source.dims,
        coords={d: np.asarray(m.model.coords[d]) for d in source.dims},
    )
    expected = (
        contributions.sel(channel="search").where(np.isfinite(ds["_target"]), 0).sum()
        / ds["_target"].sum()
    )
    np.testing.assert_allclose(evaluate(m.model, [m.model["pooled"]])[0], expected)


@pytest.mark.parametrize(
    "layout",
    [
        ("date", "region", "product", "channel"),
        ("region", "channel", "date", "product"),
        ("channel", "product", "date", "region"),
    ],
)
def test_equivalent_named_axes_rebuild_fit_and_persist(
    labeled_panel,
    layout,
    tmp_path,
    mock_pymc_sample,
):
    """Both reference identity and frozen input values respect dimension names."""
    ds, X, y = labeled_panel
    original = ds.transpose(*layout)
    m = make_model(dims=("region", "product"))
    m.build_model(original)
    m.add_contribution_prior(
        "belief", ["search"], "share", Prior("Normal", mu=0.3, sigma=0.1)
    )
    reference = copy.deepcopy(m._contribution_reference)
    expected = evaluate(
        m.model, [m.model["belief"], m.model.logp(vars=m.model.potentials)]
    )
    m.build_model(original.transpose("product", "date", "region", "channel"))
    assert m._contribution_reference == reference
    np.testing.assert_allclose(
        evaluate(m.model, [m.model["belief"], m.model.logp(vars=m.model.potentials)]),
        expected,
    )
    # Sample generation is mocked; the valid wide fit_data, graph round-trip,
    # missing-cell mask and named frozen inputs are persisted and restored for real.
    m.fit(X, y)
    path = tmp_path / "named-axes.nc"
    m.save(path)
    loaded = MMM.load(path)
    assert loaded._contribution_reference == reference
    assert len(loaded.model.potentials) == 1
    np.testing.assert_allclose(
        evaluate(
            loaded.model,
            [loaded.model["belief"], loaded.model.logp(vars=loaded.model.potentials)],
        ),
        expected,
    )
    loaded.build_model(original)
    assert loaded._contribution_reference == reference
    assert len(loaded.model.potentials) == 1
    np.testing.assert_allclose(
        evaluate(loaded.model, [loaded.model["belief"]])[0], expected[0]
    )
    assert np.isfinite(loaded.model.compile_dlogp()(loaded.model.initial_point())).all()
    with pytest.raises(ValueError, match="original training"):
        loaded.build_model(original.assign(_channel=original._channel + 1))
