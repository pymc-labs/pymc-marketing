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
"""Save, load, and specification round-trip contracts of the experimental GAM."""

import json

import numpy as np
import pandas as pd
import pymc.dims as pmd
import pytest
import xarray as xr
from pymc_extras.prior import Prior

from pymc_marketing.mmm import GeometricAdstock, LogisticSaturation, YearlyFourier
from pymc_marketing.mmm.experimental import GAM, Data, Equation, Seasonality
from pymc_marketing.mmm.experimental._serialize import spec_from_dict, spec_to_dict
from pymc_marketing.serialization import SerializationError, serialization
from pymc_marketing.special_priors import LaplacePrior
from pymc_marketing.terms import Dot, Intercept, Parameter, Sum, Transform

SAMPLE_KWARGS = {
    "draws": 30,
    "tune": 40,
    "chains": 1,
    "cores": 1,
    "random_seed": 17,
    "progressbar": False,
    "compute_convergence_checks": False,
}
DATES = pd.date_range("2025-01-06", periods=14, freq="W-MON")
FUTURE_DATES = pd.date_range(DATES[-1], periods=4, freq="W-MON")[1:]
CHANNELS = ["tv", "search"]


def _training() -> xr.Dataset:
    rng = np.random.default_rng(5)
    return xr.Dataset(
        {
            "spend": (("date", "channel"), rng.uniform(size=(len(DATES), 2))),
            "controls": (("date", "control"), rng.normal(size=(len(DATES), 1))),
            "sales": ("date", rng.normal(3, 0.3, len(DATES))),
            "orders": ("date", rng.poisson(2, len(DATES))),
            "lift": ("study", [0.5, 0.7]),
        },
        coords={
            "date": DATES,
            "channel": CHANNELS,
            "control": ["price"],
            "study": ["a", "b"],
        },
    )


def _future() -> xr.Dataset:
    rng = np.random.default_rng(6)
    return xr.Dataset(
        {
            "spend": (("date", "channel"), rng.uniform(size=(len(FUTURE_DATES), 2))),
            "controls": (("date", "control"), rng.normal(size=(len(FUTURE_DATES), 1))),
        },
        coords={"date": FUTURE_DATES, "channel": CHANNELS, "control": ["price"]},
    )


def _recipe() -> tuple[Equation, ...]:
    # Labels deliberately in the reverse of the data order.
    sigma = xr.DataArray([1.0, 2.0], dims="channel", coords={"channel": CHANNELS[::-1]})
    slope = Parameter("slope", Prior("Normal", sigma=0.5))
    contribution = (
        Data("spend")
        >> GeometricAdstock(l_max=2)
        >> LogisticSaturation(priors={"beta": Prior("HalfNormal", sigma=sigma)})
    ).named("channel_contribution")
    seasonality = Seasonality(
        YearlyFourier(
            n_order=1, prior=LaplacePrior(mu=0, b=Prior("HalfNormal"), dims="fourier")
        )
    )
    sales = Equation(
        observed="sales",
        mu=Intercept(prior=Prior("Normal", mu=3))
        + contribution.sum("channel")
        + Dot(var_name="controls", prior=Prior("Normal", dims="control"))
        + seasonality,
        likelihood=Prior("Normal", sigma=Prior("HalfNormal")),
    )
    orders = Equation(
        observed="orders",
        mu=Transform(slope * contribution.sum("channel"), pmd.math.exp),
        likelihood=Prior("Poisson"),
    )
    lift = Equation(observed="lift", mu=slope, likelihood=Prior("Normal", sigma=0.2))
    return sales, orders, lift


@pytest.fixture(scope="module")
def fitted() -> GAM:
    gam = GAM(*_recipe())
    gam.fit(_training(), **SAMPLE_KWARGS)
    return gam


def test_spec_round_trip_rebuilds_the_same_joint_model():
    roots = _recipe()
    restored = spec_from_dict(json.loads(json.dumps(spec_to_dict(roots))))

    original = GAM(*roots).build_model(_training())
    rebuilt = GAM(*restored).build_model(_training())

    assert rebuilt.named_vars_to_dims == original.named_vars_to_dims
    assert [rv.name for rv in rebuilt.free_RVs] == [rv.name for rv in original.free_RVs]
    rng = np.random.default_rng(0)
    point = {
        name: value + rng.normal(scale=0.3, size=np.shape(value))
        for name, value in original.initial_point().items()
    }
    np.testing.assert_allclose(
        rebuilt.compile_logp()(point), original.compile_logp()(point), rtol=1e-12
    )


@pytest.mark.parametrize("suffix", [".zarr", ".zarr.zip"])
def test_loaded_model_forecasts_like_the_fitted_one(fitted, tmp_path, suffix):
    path = tmp_path / f"model{suffix}"
    fitted.save(path)
    loaded = GAM.load(path)

    options = {
        "var_names": ["sales", "channel_contribution"],
        "random_seed": 4,
        "progressbar": False,
    }
    xr.testing.assert_allclose(
        loaded.sample_posterior_predictive(_future(), **options),
        fitted.sample_posterior_predictive(_future(), **options),
    )
    assert sorted(loaded.idata.children) == [
        "observed_data",
        "posterior",
        "sample_stats",
    ]


def test_saved_spec_is_plain_json_in_the_root_metadata(fitted, tmp_path):
    path = tmp_path / "model.zarr"
    fitted.save(path)

    attributes = json.loads((path / "zarr.json").read_text())["attributes"]

    assert attributes["spec"]["format"] == "pymc_marketing.experimental.GAM/1"
    assert {"spec", "versions", "logp_check"} <= attributes.keys()


def _change_intercept_prior(spec):
    intercept = next(
        node
        for node in spec["nodes"].values()
        if node["__type__"].endswith("terms.Intercept")
    )
    intercept["fields"]["prior"]["$prior"]["parameters"]["mu"] = 5


def _change_format(spec):
    spec["format"] = "pymc_marketing.experimental.GAM/99"


@pytest.mark.parametrize(
    ("tamper", "error", "match"),
    [
        pytest.param(
            _change_intercept_prior,
            ValueError,
            "does not reproduce the saved log-density",
            id="model-changed",
        ),
        pytest.param(
            _change_format,
            SerializationError,
            "Unsupported saved model format",
            id="unknown-format",
        ),
    ],
)
def test_load_rejects_files_that_do_not_rebuild_the_saved_model(
    fitted, tmp_path, tamper, error, match
):
    path = tmp_path / "model.zarr"
    fitted.save(path)
    metadata = path / "zarr.json"
    content = json.loads(metadata.read_text())
    tamper(content["attributes"]["spec"])
    metadata.write_text(json.dumps(content))

    with pytest.raises(error, match=match):
        GAM.load(path)


def test_load_check_can_be_skipped(fitted, tmp_path):
    path = tmp_path / "model.zarr"
    fitted.save(path)
    metadata = path / "zarr.json"
    content = json.loads(metadata.read_text())
    _change_intercept_prior(content["attributes"]["spec"])
    metadata.write_text(json.dumps(content))

    assert GAM.load(path, check=False).idata is not None


def test_save_rejects_unsaveable_terms_and_paths(fitted, tmp_path):
    with pytest.raises(SerializationError, match="not serializable"):
        spec_to_dict(
            [Equation(observed="y", mu=Transform(Parameter("a"), lambda value: value))]
        )
    with pytest.raises(SerializationError, match=r"GAM\.save"):
        serialization.serialize(Sum([Data("spend")]))
    with pytest.raises(ValueError, match="zarr"):
        fitted.save(tmp_path / "model.nc")
