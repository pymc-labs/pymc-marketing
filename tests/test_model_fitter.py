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
"""Tests for the consolidated `ModelFitter` mixin."""

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

from pymc_marketing.model_builder import ModelBuilder, _approx_fit_parameters
from pymc_marketing.version import __version__

ALL_METHODS = ["mcmc", "map", "demz", "advi", "fullrank_advi"]


class FitterModel(ModelBuilder):
    """Minimal `ModelBuilder` exercising the shared fitting pipeline.

    Holds a deterministic (``double_a``) so the free-RV/recompute split is observable.
    """

    _model_type = "fitter_test_model"
    version = "0.1"

    def __init__(self, data=None, model_config=None, sampler_config=None):
        super().__init__(model_config=model_config, sampler_config=sampler_config)
        self.data = data

    @property
    def default_model_config(self) -> dict:
        return {"a_scale": 5.0}

    @property
    def default_sampler_config(self) -> dict:
        return {"draws": 20, "tune": 20, "chains": 1, "progressbar": False}

    @property
    def _serializable_model_config(self) -> dict:
        return self.model_config

    def build_model(self, data=None) -> None:
        data = self.data if data is None else data
        with pm.Model(coords={"obs": np.arange(len(data))}) as self.model:
            a = pm.Normal("a", 0, self.model_config["a_scale"])
            pm.Deterministic("double_a", a * 2)
            sigma = pm.HalfNormal("sigma", 1)
            pm.Normal("y", a, sigma, observed=data["y"].values, dims="obs")

    def build_from_idata(self, idata: xr.DataTree) -> None:
        self.data = idata.fit_data.dataset.to_dataframe()
        self.build_model()


@pytest.fixture(scope="module")
def toy_data() -> pd.DataFrame:
    rng = np.random.default_rng(sum(map(ord, "model fitter")))
    return pd.DataFrame({"y": rng.normal(2.0, 1.0, size=25)})


def _fit_kwargs(method: str) -> dict:
    """Sampler settings small enough to keep every backend fast."""
    if method in ("advi", "fullrank_advi"):
        return {"n": 100, "sample_kwargs": {"draws": 10}}
    if method == "map":
        return {"progressbar": False}
    return {}


@pytest.mark.parametrize("method", ALL_METHODS)
def test_fit_produces_consistent_idata(toy_data, method) -> None:
    """Every sampler yields the same idata shape: posterior, fit_data, attrs, flag."""
    model = FitterModel(data=toy_data)

    idata = model.fit(method=method, random_seed=42, **_fit_kwargs(method))

    assert isinstance(idata, xr.DataTree)
    assert "/posterior" in idata.groups
    assert "/fit_data" in idata.groups
    assert idata["/posterior"].attrs["pymc_marketing_version"] == __version__
    assert set(idata.attrs) >= {
        "id",
        "model_type",
        "version",
        "sampler_config",
        "model_config",
    }
    assert model.is_fitted_


@pytest.mark.parametrize("method", ALL_METHODS)
def test_fit_always_returns_deterministics(toy_data, method) -> None:
    """Deterministics must survive every path, whether recomputed or sampled inline."""
    model = FitterModel(data=toy_data)

    idata = model.fit(method=method, random_seed=42, **_fit_kwargs(method))

    posterior = idata["/posterior"].to_dataset()
    assert {"a", "sigma", "double_a"} <= set(posterior.data_vars)
    np.testing.assert_allclose(
        posterior["double_a"].values, posterior["a"].values * 2, rtol=1e-5
    )


def test_recompute_deterministics_matches_inline(toy_data) -> None:
    """Deferring deterministics to a vectorized recompute is value-preserving."""
    recomputed = FitterModel(data=toy_data)
    recomputed.fit(random_seed=42)

    inline = FitterModel(data=toy_data)
    inline._recompute_deterministics = False
    inline.fit(random_seed=42)

    left = recomputed.idata["/posterior"].to_dataset()
    right = inline.idata["/posterior"].to_dataset()

    assert set(left.data_vars) == set(right.data_vars)
    for name in left.data_vars:
        np.testing.assert_allclose(left[name].values, right[name].values, rtol=1e-6)


def test_fit_unknown_method_raises(toy_data) -> None:
    model = FitterModel(data=toy_data)

    match = (
        r"Fit method options are \['mcmc', 'map', 'demz', 'advi', "
        r"'fullrank_advi'\], got: wrong_method"
    )
    with pytest.raises(ValueError, match=match):
        model.fit(method="wrong_method")


def test_refit_replaces_posterior_and_keeps_prior(toy_data) -> None:
    """Refitting merges into the existing tree rather than discarding other groups."""
    model = FitterModel(data=toy_data)
    model.build_model()
    with model.model:
        prior = pm.sample_prior_predictive(draws=5, random_seed=42)
    model.idata = prior

    model.fit(random_seed=42)

    assert "/prior" in model.idata.groups
    assert "/posterior" in model.idata.groups
    assert "/fit_data" in model.idata.groups


def test_refit_with_different_method_drops_stale_groups(toy_data) -> None:
    """Groups derived from the old posterior must not survive a MAP refit."""
    model = FitterModel(data=toy_data)
    model.fit(random_seed=42)
    assert model.idata["/sample_stats"].to_dataset().sizes["draw"] == 20
    model.idata["/posterior_predictive"] = model.idata["/posterior"].to_dataset()

    model.fit(method="map", progressbar=False)

    assert "/posterior_predictive" not in model.idata.groups
    assert model.idata["/posterior"].to_dataset().sizes["draw"] == 1
    assert "draw" not in model.idata["/sample_stats"].to_dataset().sizes


def test_base_fit_rejects_unrouted_data(toy_data) -> None:
    """The base pipeline has nowhere to send `data`, so it must refuse it loudly."""
    model = FitterModel(data=toy_data)

    with pytest.raises(NotImplementedError, match="does not accept `data`"):
        model.fit(data=toy_data)


def test_fit_does_not_mutate_sampler_config(toy_data) -> None:
    sampler_config = {"draws": 20, "tune": 20, "chains": 1, "progressbar": False}
    model = FitterModel(data=toy_data, sampler_config=dict(sampler_config))

    model.fit(random_seed=42)

    assert model.sampler_config == sampler_config


def test_fit_data_group_round_trips(toy_data) -> None:
    model = FitterModel(data=toy_data)
    model.fit(random_seed=42)

    stored = model.idata.fit_data.dataset.to_dataframe()
    pd.testing.assert_series_equal(stored["y"], toy_data["y"], check_index=False)


def test_map_seed_accepts_numpy_generator(toy_data) -> None:
    """Generator seeds must be converted for `pm.find_MAP`, not silently dropped."""
    model = FitterModel(data=toy_data)

    idata = model.fit(
        method="map", random_seed=np.random.default_rng(42), progressbar=False
    )

    assert idata["/posterior"].to_dataset().sizes["draw"] == 1


def test_map_seed_warns_on_unsupported_type(toy_data) -> None:
    model = FitterModel(data=toy_data)

    with pytest.warns(UserWarning, match="not supported with method='map'"):
        model.fit(method="map", random_seed=[1, 2], progressbar=False)


def test_advi_config_seed_makes_draws_reproducible(toy_data) -> None:
    """A seed in sampler_config must reach `Approximation.sample`, not just `pm.fit`."""

    def fit_once() -> xr.Dataset:
        model = FitterModel(
            data=toy_data,
            sampler_config={"chains": 1, "progressbar": False, "random_seed": 42},
        )
        model.fit(method="advi", n=100, sample_kwargs={"draws": 10})
        return model.idata["/posterior"].to_dataset()

    left, right = fit_once(), fit_once()

    for name in left.data_vars:
        np.testing.assert_allclose(left[name].values, right[name].values)


def test_fit_data_group_warning_is_suppressed(toy_data, recwarn) -> None:
    model = FitterModel(data=toy_data)

    model.fit(random_seed=42)

    assert not [w for w in recwarn if "fit_data is not defined" in str(w.message)]


def test_advi_reports_unusable_kwargs(toy_data) -> None:
    """MCMC-only keys are dropped from the VI path, but never silently."""
    model = FitterModel(data=toy_data)

    with pytest.warns(UserWarning, match=r"not accepted by 'advi'.*'tune'"):
        model.fit(method="advi", n=100, sample_kwargs={"draws": 10})


def test_demz_drops_nuts_only_kwargs(toy_data) -> None:
    """`target_accept` in sampler_config must not break the gradient-free path.

    ``pm.sample`` routes NUTS-only keys into ``step_kwargs`` and raises when an
    explicit ``step`` is also supplied, so ``demz`` has to strip them.
    """
    model = FitterModel(
        data=toy_data,
        sampler_config={
            "draws": 20,
            "tune": 20,
            "chains": 1,
            "progressbar": False,
            "target_accept": 0.95,
        },
    )

    with pytest.warns(UserWarning, match=r"only apply to NUTS.*'target_accept'"):
        idata = model.fit(method="demz", random_seed=42)

    assert "/posterior" in idata.groups


def test_advi_warns_on_multiple_chains(toy_data) -> None:
    model = FitterModel(data=toy_data, sampler_config={"chains": 2})

    with pytest.warns(
        UserWarning, match="The 'chains' parameter must be 1 with 'advi'"
    ):
        model.fit(method="advi", n=100, sample_kwargs={"draws": 10})


def test_approx_fit_parameters_are_derived() -> None:
    """The VI kwarg filter is introspected, not hard-coded."""
    allowed = _approx_fit_parameters()

    # spread across pymc.fit, Inference.fit and ObjectiveFunction.step_function
    assert {"n", "random_seed", "start", "score", "obj_optimizer"} <= allowed
    assert "tune" not in allowed
    assert "chains" not in allowed
