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
from pymc_extras.prior import Prior

from pymc_marketing.mmm import MMM
from tests.mmm.contribution_factors_helpers import data as data
from tests.mmm.contribution_factors_helpers import (
    evaluate,
    make_model,
)
from tests.mmm.contribution_factors_helpers import labeled_panel as labeled_panel


def test_frozen_reference_prediction_optimization_and_rebuild(data):
    X, y = data
    m = make_model()
    m.build_model(X, y)
    m.add_contribution_factor(
        "belief", ["search"], "share", Prior("Normal", mu=0.3, sigma=0.1)
    )
    before = evaluate(
        m.model, [m.model["belief"], m.model.logp(vars=m.model.potentials)]
    )
    reference = copy.deepcopy(m._contribution_factors.fingerprint)
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
    assert m._contribution_factors.fingerprint == reference
    # Adding another component after prediction must still use training inputs.
    m.add_contribution_factor(
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
    "prior", [Prior("Normal", mu=0.3, sigma=0.1), Prior("LogNormal", mu=-1, sigma=0.5)]
)
def test_save_load_restore_original_missingness(data, tmp_path, prior):
    X, y = data
    m = make_model()
    m.build_model(X, y)
    m.add_contribution_factor("belief", ["search"], "share", prior)
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
    assert loaded._contribution_factors.specs == m._contribution_factors.specs
    assert (
        loaded._contribution_factors.fingerprint == m._contribution_factors.fingerprint
    )
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
    assert not bool(loaded.idata.contribution_factor_data.observed_mask.isel(date=2))


@pytest.mark.parametrize(
    "layout",
    [
        ("date", "region", "product", "channel"),
        ("region", "channel", "date", "product"),
        ("channel", "product", "date", "region"),
    ],
)
@pytest.mark.parametrize("permute_labels", [False, True])
def test_equivalent_named_axes_rebuild_fit_and_persist(
    labeled_panel,
    layout,
    permute_labels,
    tmp_path,
    mock_pymc_sample,
):
    """Both reference identity and frozen input values respect dimension names."""
    ds, X, y = labeled_panel
    if permute_labels:
        ds = ds.sel(region=["B", "A"], product=["q", "p"])
    original = ds.transpose(*layout)
    m = make_model(dims=("region", "product"))
    m.build_model(original)
    m.add_contribution_factor(
        "belief", ["search"], "share", Prior("Normal", mu=0.3, sigma=0.1)
    )
    reference = copy.deepcopy(m._contribution_factors.fingerprint)
    expected = evaluate(
        m.model, [m.model["belief"], m.model.logp(vars=m.model.potentials)]
    )
    m.build_model(original.transpose("product", "date", "region", "channel"))
    assert m._contribution_factors.fingerprint == reference
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
    assert loaded._contribution_factors.fingerprint == reference
    assert len(loaded.model.potentials) == 1
    np.testing.assert_allclose(
        evaluate(
            loaded.model,
            [loaded.model["belief"], loaded.model.logp(vars=loaded.model.potentials)],
        ),
        expected,
    )
    loaded.build_model(original)
    assert loaded._contribution_factors.fingerprint == reference
    assert len(loaded.model.potentials) == 1
    np.testing.assert_allclose(
        evaluate(loaded.model, [loaded.model["belief"]])[0], expected[0]
    )
    assert np.isfinite(loaded.model.compile_dlogp()(loaded.model.initial_point())).all()
    with pytest.raises(ValueError, match="original training"):
        loaded.build_model(original.assign(_channel=original._channel + 1))


def test_post_fit_addition_rejected(data, mock_pymc_sample):
    m = make_model()
    m.fit(*data)
    names = set(m.model.named_vars)
    with pytest.raises(ValueError, match="before fitting"):
        m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    assert set(m.model.named_vars) == names


def test_time_slice_cv_rejects_rebinding_historical_factors(data):
    from pymc_marketing.mmm import TimeSliceCrossValidator

    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    cv = TimeSliceCrossValidator(n_init=3, forecast_horizon=1, date_column="date")
    with pytest.raises(ValueError, match="time-slice cross-validation"):
        cv.run(*data, mmm=m)
    assert "belief" in m.model.named_vars


@pytest.mark.parametrize("corruption", ["missing_mask", "unknown_version"])
def test_failed_load_restoration_preserves_existing_model(data, corruption):
    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    model = m.model
    before = evaluate(model, [model["belief"], model.logp()])
    broken = xr.DataTree.from_dict({"fit_data": m.create_fit_data(*data)})
    broken.attrs.update(m.create_idata_attrs())
    error: type[Exception] = AttributeError
    match = "contribution_factor_data"
    if corruption == "unknown_version":
        metadata = json.loads(broken.attrs["contribution_factors"])
        metadata["version"] = 2
        broken.attrs["contribution_factors"] = json.dumps(metadata)
        error, match = (
            ValueError,
            "Unsupported contribution-factor serialization version",
        )
    with pytest.raises(error, match=match):
        m.build_from_idata(broken)
    assert m.model is model
    assert m.idata is None
    np.testing.assert_allclose(
        evaluate(m.model, [m.model["belief"], m.model.logp()]), before
    )


def test_failed_factor_rebuild_isolates_mutable_configuration(data, monkeypatch):
    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    model = m.model
    effects = m.mu_effects
    configuration = copy.deepcopy(m.model_config)
    original = MMM._build_graph

    def broken(working, X, y):
        original(working, X, y)
        working.mu_effects.append(object())
        working.model_config["saturation_beta"] = Prior("Normal")
        raise RuntimeError("reject after in-place mutations")

    monkeypatch.setattr(MMM, "_build_graph", broken)
    with pytest.raises(RuntimeError, match="in-place mutations"):
        m.build_model(*data)
    assert m.model is model
    assert m.mu_effects is effects
    assert m.mu_effects == []
    assert m.model_config == configuration


@pytest.mark.parametrize("family", ["LogNormal", "HalfNormal"])
def test_positive_factor_rebuild_rejects_new_signed_effects(data, family):
    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior(family, sigma=0.1))
    model = m.model
    m.saturation.update_priors(
        {"saturation_beta": Prior("Normal", mu=0.3, sigma=0.5, dims="channel")}
    )
    with pytest.raises(ValueError, match="nonnegative support"):
        m.build_model(*data)
    assert m.model is model
    assert len(m._contribution_factors.specs) == 1


@pytest.mark.parametrize("locked", [False, True])
def test_factor_rebuild_preserves_user_components(data, locked):
    from threading import Lock
    from typing import Any

    from pydantic import Field

    from pymc_marketing.mmm import LinearTrend
    from pymc_marketing.mmm.additive_effect import LinearTrendEffect

    class ResourceTrend(LinearTrendEffect):
        resource: Any = Field(default=None, exclude=True)

    effect = ResourceTrend(
        trend=LinearTrend(n_changepoints=2),
        prefix="trend",
        resource=Lock() if locked else None,
    )
    m = make_model()
    m.add_mu_effect(effect)
    m.build_model(*data)
    config, effects = m.model_config, m.mu_effects
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    m.build_model(*data)
    assert m.mu_effects is effects
    assert m.mu_effects[0] is effect
    assert m.model_config is config
    assert np.isfinite(m.model.compile_dlogp()(m.model.initial_point())).all()


def test_factor_mask_order_matches_denominator_on_load(
    labeled_panel, tmp_path, mock_pymc_sample
):
    ds, X, y = labeled_panel
    m = make_model(dims=("region", "product"))
    m.build_model(ds.sel(region=["B", "A"], product=["q", "p"]))
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    expected = evaluate(m.model, [m.model["belief"]])[0]
    m.fit(X, y)
    path = tmp_path / "without-posterior.nc"
    m.save(str(path), groups=["fit_data", "contribution_factor_data"])
    loaded = MMM.load(str(path))
    np.testing.assert_allclose(
        evaluate(loaded.model, [loaded.model["belief"]])[0], expected
    )


def test_factor_rebuild_preserves_hsgp_registration_on_failure(data, monkeypatch):
    from pymc_marketing.mmm.hsgp import SoftPlusHSGP

    factory = SoftPlusHSGP(m=3, L=5, dims="date", eta=1, ls=2)
    m = make_model(time_varying_media=factory)
    m.build_model(*data)
    m.add_contribution_factor(
        "belief", ["search"], "share", Prior("HalfNormal", sigma=0.5)
    )
    old_model, old_data = m.model, factory.X
    original = MMM._build_graph

    def broken(working, X, y):
        original(working, X, y)
        raise RuntimeError("failed staged registration")

    with monkeypatch.context() as context:
        context.setattr(MMM, "_build_graph", broken)
        with pytest.raises(RuntimeError, match="failed staged registration"):
            m.build_model(*data)
    assert m.model is old_model
    assert m.time_varying_media is factory
    assert factory.X is old_data
    m.build_model(*data)
    assert m.time_varying_media is factory
    assert factory.X is m.model["time_index"]


def test_factor_rebuild_invalidates_compiled_evaluator_cache(data):
    m = make_model()
    m.build_model(*data)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    previous = m._incrementality_evaluator_cache()
    m.build_model(*data)
    assert m._incrementality_evaluator_cache() is not previous


def test_factor_reference_with_supplementary_effect_data(
    data, tmp_path, mock_pymc_sample
):
    from unittest.mock import patch

    from pymc_marketing.mmm.additive_effect import ControlMuEffect

    X, y = data
    X = X.assign(external=np.arange(len(X), dtype=float))
    m = make_model()
    m.add_mu_effect(ControlMuEffect(data_vars=["external"], prefix="macro"))
    m.build_model(X, y)
    m.add_contribution_factor("belief", ["search"], "share", Prior("Normal"))
    expected = evaluate(m.model, [m.model["belief"]])[0]
    m.build_model(X, y)
    np.testing.assert_allclose(evaluate(m.model, [m.model["belief"]])[0], expected)
    with pytest.raises(ValueError, match="original training cells"):
        m.build_model(X.assign(external=100.0), y)
    m.fit(X, y)
    path = tmp_path / "supplementary-data.nc"
    m.save(str(path))
    loaded = MMM.load(str(path))
    np.testing.assert_allclose(
        evaluate(loaded.model, [loaded.model["belief"]])[0], expected
    )
    future = X.assign(date=pd.date_range("2026-01-01", periods=len(X)), external=100.0)
    loaded.sample_posterior_predictive(
        future, clone_model=False, extend_idata=False, random_seed=123
    )

    def sample(**kwargs):
        model = pm.modelcontext(None)
        np.testing.assert_allclose(
            evaluate(model, [model["external"]])[0], X.external.to_numpy()
        )
        np.testing.assert_allclose(evaluate(model, [model["belief"]])[0], expected)
        return xr.DataTree.from_dict({"posterior": xr.Dataset()})

    with patch("pymc.sample", side_effect=sample):
        loaded.sample_effective_prior()
