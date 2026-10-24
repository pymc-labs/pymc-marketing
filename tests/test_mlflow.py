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
import functools
import json
import logging
import os
import sys
import tempfile
from collections import namedtuple
from collections.abc import Callable, Iterator
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import mlflow
import mlflow.artifacts
import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytest
import xarray as xr
from mlflow.client import MlflowClient
from pymc.backends.base import IBaseTrace
from pymc.exceptions import SamplingError
from pymc_extras.prior import Prior

import pymc_marketing.mlflow as pmm_mlflow
from pymc_marketing.bass import BassModel
from pymc_marketing.clv import BetaGeoModel
from pymc_marketing.clv.models.basic import CLVModel
from pymc_marketing.data.idata.io import load_idata
from pymc_marketing.mlflow import (
    _idata_artifact_name,
    autolog,
    create_log_callback,
    load_mmm,
    load_mmm_from_run,
    log_error,
    log_inference_data,
    log_likelihood_type,
    log_mmm,
    log_mmm_evaluation_metrics,
    log_model_graph,
    log_sample_diagnostics,
    log_versions,
)
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_marketing.version import __version__

seed = sum(map(ord, "mlflow-with-pymc"))
rng = np.random.default_rng(seed)


def _unwrap_autolog() -> None:
    """Remove the wrappers ``autolog`` installed on every patched callable."""
    pm.sample = pmm_mlflow._without_autolog_wrappers(pm.sample)
    pm.find_MAP = pmm_mlflow._without_autolog_wrappers(pm.find_MAP)
    MMM.fit = pmm_mlflow._without_autolog_wrappers(MMM.fit)
    CLVModel.fit = pmm_mlflow._without_autolog_wrappers(CLVModel.fit)
    BassModel.fit = pmm_mlflow._without_autolog_wrappers(BassModel.fit)


@pytest.fixture(scope="function", autouse=True)
def idata_format(request) -> Iterator[str]:
    """Enable autologging for each test and yield the inference-data format in use."""
    fmt = getattr(request, "param", "netcdf")
    mlflow.set_tracking_uri(uri=f"sqlite:///{Path('mlruns.db').resolve()}")
    autolog(idata_format=fmt)

    yield fmt

    _unwrap_autolog()


def _filesystem_is_case_insensitive() -> bool:
    """Probe the working directory: autolog writes its Zarr artifact there."""
    with tempfile.TemporaryDirectory(dir=".") as tmp:
        (Path(tmp) / "CaseProbe").touch()
        return (Path(tmp) / "caseprobe").exists()


CASE_INSENSITIVE_FS = _filesystem_is_case_insensitive()


def _wrapper_depth(func: Callable) -> int:
    depth = 0
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
        depth += 1
    return depth


@pytest.fixture(scope="module")
def model_with_likelihood() -> pm.Model:
    n_obs = 15

    data = rng.normal(loc=5, scale=2, size=n_obs)

    coords = {
        "obs_id": np.arange(n_obs),
    }
    with pm.Model(coords=coords) as model:
        mu = pm.Normal("mu", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)

        pm.Normal("obs", mu=mu, sigma=sigma, observed=data)

    return model


@pytest.fixture(scope="module")
def model_with_data_in_likelihood() -> pm.Model:
    n_obs = 15

    data = rng.normal(loc=5, scale=2, size=n_obs)

    coords = {
        "obs_id": np.arange(n_obs),
    }
    with pm.Model(coords=coords) as model:
        mu = pm.Normal("mu", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)

        target = pm.Data("target", data, dims="obs_id")
        pm.Normal("obs", mu=mu, sigma=sigma, observed=target, dims="obs_id")

    return model


@pytest.fixture(scope="module")
def no_input_model() -> pm.Model:
    with pm.Model() as model:
        pm.Normal("mu")
        pm.HalfNormal("sigma")

    return model


@pytest.fixture(scope="module")
def multi_likelihood_model() -> pm.Model:
    n_obs = 15

    mu = 10
    scale = 2
    data1 = pm.draw(pm.Normal.dist(mu=mu, sigma=scale, size=n_obs), random_seed=rng)
    data2 = pm.draw(pm.Gamma.dist(alpha=mu, beta=scale, size=n_obs), random_seed=rng)

    coords = {
        "obs_id": np.arange(n_obs),
    }
    with pm.Model(coords=coords) as model:
        mu = pm.Normal("mu", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)

        pm.Normal("obs1", mu=mu, sigma=sigma, observed=data1)
        pm.Gamma("obs2", mu=mu, sigma=sigma, observed=data2)

    return model


RunData = namedtuple(
    "RunData",
    ["inputs", "params", "metrics", "tags", "artifacts"],
)


def get_run_data(run_id) -> RunData:
    # Adapted from mlflow tests for sklearn autolog
    client = MlflowClient()
    run = client.get_run(run_id)
    data = run.data
    # Ignore tags mlflow logs by default (e.g. "mlflow.user")
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id)]
    inputs = [inp for inp in run.inputs.dataset_inputs]

    return RunData(
        inputs=inputs,
        params=data.params,
        metrics=data.metrics,
        tags=tags,
        artifacts=artifacts,
    )


def basic_logging_checks(run_data: RunData) -> None:
    assert len(run_data.params) > 0
    assert len(run_data.metrics) > 0
    assert run_data.tags == {}
    assert len(run_data.artifacts) > 0


def test_log_with_data_in_likelihood(model_with_data_in_likelihood) -> None:
    mlflow.set_experiment("pymc-marketing-test-suite-only-target")
    with mlflow.start_run() as run:
        pm.sample(
            model=model_with_data_in_likelihood,
            chains=1,
            draws=25,
            tune=10,
        )

    run_id = run.info.run_id
    run_data = get_run_data(run_id)

    basic_logging_checks(run_data)

    inputs = run_data.inputs

    assert len(inputs) == 1
    profile = json.loads(inputs[0].dataset.profile)

    expected_feature_shape = {}
    expected_target_shape = {"obs": [15]}

    assert profile["features_shape"] == expected_feature_shape
    assert profile["targets_shape"] == expected_target_shape

    assert run_data.params["likelihood"] == "Normal"
    assert run_data.params["n_free_RVs"] == "2"
    assert run_data.params["n_observed_RVs"] == "1"
    assert run_data.params["n_deterministics"] == "0"
    assert run_data.params["n_potentials"] == "0"


def no_input_model_checks(run_data: RunData) -> None:
    assert run_data.inputs == []


def test_log_data_no_data(no_input_model) -> None:
    mlflow.set_experiment("pymc-marketing-test-suite-no-data")
    with mlflow.start_run() as run:
        pm.sample(
            model=no_input_model,
            chains=1,
            draws=25,
            tune=10,
        )

    run_id = run.info.run_id
    run_data = get_run_data(run_id)

    no_input_model_checks(run_data)
    basic_logging_checks(run_data)


def test_run_id_attached_to_idata(model_with_likelihood, tmp_path) -> None:
    mlflow.set_experiment("pymc-marketing-test-suite-run-id-attr")
    with mlflow.start_run() as run:
        idata = pm.sample(
            model=model_with_likelihood,
            chains=1,
            draws=25,
            tune=10,
        )

    assert idata.attrs["mlflow_run_id"] == run.info.run_id

    save_path = tmp_path / "idata.nc"
    idata.to_netcdf(str(save_path))
    reloaded = xr.open_datatree(str(save_path))
    assert reloaded.attrs["mlflow_run_id"] == run.info.run_id


def test_attach_run_id_no_active_run_is_noop() -> None:
    assert mlflow.active_run() is None
    idata = xr.DataTree.from_dict({})
    pmm_mlflow._attach_run_id(idata)
    assert "mlflow_run_id" not in idata.attrs


def test_multi_likelihood_type(multi_likelihood_model) -> None:
    mlflow.set_experiment("pymc-marketing-test-suite-multi-likelihood")
    with mlflow.start_run() as run:
        log_likelihood_type(multi_likelihood_model)

    run_id = run.info.run_id
    run_data = get_run_data(run_id)

    assert run_data.params == {
        "observed_RVs_types": "['Normal', 'Gamma']",
    }


def test_dims_censored_likelihood_type() -> None:
    """`pymc.dims` builds a censored variable as a clip, not as a CensoredRV.

    There is no distribution name on the op to read, so the name has to come
    from the shape of the graph instead.
    """
    coords = {"T": np.arange(3)}
    with pm.Model(coords=coords) as model:
        pmd.Censored(
            "y",
            pmd.Normal.dist(mu=0, sigma=1),
            lower=0,
            upper=None,
            dims=("T",),
            observed=pmd.as_xtensor(np.ones(3), dims=("T",)),
        )

    mlflow.set_experiment("pymc-marketing-test-suite-dims-censored")
    with mlflow.start_run() as run:
        log_likelihood_type(model)

    assert get_run_data(run.info.run_id).params == {"likelihood": "Censored"}


@pytest.mark.parametrize(
    "to_patch, side_effect, expected_info_message",
    [
        (
            "pymc.model_to_graphviz",
            ImportError("No module named 'graphviz'"),
            "Unable to render the model graph. Please install the graphviz package. No module named 'graphviz'",
        ),
        (
            "graphviz.graphs.Digraph.render",
            Exception("Unknown error occurred"),
            "Unable to render the model graph. Unknown error occurred",
        ),
        (
            "pymc.model_to_graphviz",
            ValueError("lam < 0 or lam contains NaNs"),
            "Unable to render the model graph. lam < 0 or lam contains NaNs",
        ),
    ],
    ids=["no_graphviz", "render_error", "graph_creation_error"],
)
def test_log_model_graph_no_graphviz(
    caplog,
    mocker,
    model_with_likelihood,
    to_patch,
    side_effect,
    expected_info_message,
) -> None:
    mocker.patch(
        to_patch,
        side_effect=side_effect,
    )
    with mlflow.start_run() as run:
        with caplog.at_level(logging.INFO, logger="pymc_marketing.mlflow"):
            log_model_graph(model_with_likelihood, "model_graph")

    # Only inspect records emitted by pymc-marketing itself. caplog's handler is
    # attached to the root logger, so it also captures unrelated INFO logs from
    # third-party libraries (e.g. MLflow's "Creating initial MLflow database
    # tables..." emitted on first backend-store creation), which would otherwise
    # make this assertion order-dependent and flaky.
    messages = [
        record.message
        for record in caplog.records
        if record.name == "pymc_marketing.mlflow"
    ]
    assert messages == [
        expected_info_message,
    ]

    run_id = run.info.run_id
    artifacts = get_run_data(run_id)[-1]

    assert artifacts == []


def metric_checks(metrics, nuts_sampler) -> None:
    assert metrics["total_divergences"] >= 0.0
    if nuts_sampler not in ["numpyro", "nutpie", "blackjax"]:
        assert metrics["sampling_time"] >= 0.0
        assert metrics["time_per_draw"] >= 0.0


def param_checks(params, draws: int, chains: int, tune: int, nuts_sampler: str) -> None:
    assert params["draws"] == str(draws)
    assert params["chains"] == str(chains)
    assert params["posterior_samples"] == str(draws * chains)

    if nuts_sampler not in ["numpyro", "blackjax"]:
        assert params["inference_library"] == nuts_sampler

    assert params["tuning_steps"] == str(tune)
    assert params["tuning_samples"] == str(tune * chains)

    assert params["pymc_marketing_version"] == __version__

    other_keys = ["pymc_version", "xarray_version"]
    if nuts_sampler not in ["numpyro", "blackjax"]:
        other_keys.extend(["inference_library_version"])

    for other_key in other_keys:
        assert other_key in params


@pytest.mark.parametrize(
    "nuts_sampler",
    [
        "pymc",
        "numpyro",
        "nutpie",
        "blackjax",
    ],
)
def test_autolog_pymc_model(model_with_likelihood, nuts_sampler) -> None:
    mlflow.set_experiment("pymc-marketing-test-suite-pymc-model")
    with mlflow.start_run() as run:
        draws = 30
        tune = 25
        chains = 2
        pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            model=model_with_likelihood,
            nuts_sampler=nuts_sampler,
        )

    assert mlflow.active_run() is None

    run_id = run.info.run_id
    inputs, params, metrics, tags, artifacts = get_run_data(run_id)

    param_checks(
        params=params,
        draws=draws,
        chains=chains,
        tune=tune,
        nuts_sampler=nuts_sampler,
    )

    assert params["n_free_RVs"] == "2"
    assert params["n_observed_RVs"] == "1"
    assert params["n_deterministics"] == "0"
    assert params["n_potentials"] == "0"
    assert params["likelihood"] == "Normal"

    metric_checks(metrics, nuts_sampler)

    assert tags == {}
    assert artifacts == [
        "coords.json",
        "model_graph.pdf",
        "model_repr.txt",
        "summary.html",
    ]

    assert len(inputs) == 1


@pytest.fixture(scope="module")
def bad_starting_point_model() -> pm.Model:
    data = [-5, -3, -1, 0, 1]

    coords = {"idx": range(len(data))}
    with pm.Model(coords=coords) as model:
        alpha = pm.HalfNormal("alpha")
        beta = pm.HalfNormal("beta")

        pm.Gamma("obs", alpha=alpha, beta=beta, observed=data, dims="idx")

    return model


@pytest.mark.parametrize(
    "nuts_sampler",
    [
        "pymc",
        "numpyro",
        "nutpie",
        "blackjax",
    ],
)
def test_sample_error_logged(bad_starting_point_model, nuts_sampler: str) -> None:
    mlflow.set_experiment("pymc-marketing-test-suite-error-model")
    with mlflow.start_run() as run:
        draws = 30
        tune = 25
        chains = 2
        try:
            pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                model=bad_starting_point_model,
                nuts_sampler=nuts_sampler,
            )
        except Exception as e:
            error = RuntimeError if nuts_sampler == "nutpie" else SamplingError
            assert isinstance(e, error)

    assert mlflow.active_run() is None

    run_id = run.info.run_id
    *_, artifacts = get_run_data(run_id)

    assert "sample-error.txt" in artifacts


@pytest.fixture(scope="module")
def generate_data():
    def _generate_data(date_data: pd.DatetimeIndex) -> pd.DataFrame:
        n: int = date_data.size

        return pd.DataFrame(
            data={
                "date": date_data,
                "channel_1": rng.integers(low=0, high=400, size=n),
                "channel_2": rng.integers(low=0, high=50, size=n),
                "control_1": rng.gamma(shape=1000, scale=500, size=n),
                "control_2": rng.gamma(shape=100, scale=5, size=n),
                "other_column_1": rng.integers(low=0, high=100, size=n),
                "other_column_2": rng.normal(loc=0, scale=1, size=n),
            }
        )

    return _generate_data


@pytest.fixture(scope="module")
def toy_X(generate_data) -> pd.DataFrame:
    date_data: pd.DatetimeIndex = pd.date_range(
        start="2019-06-01", end="2021-12-31", freq="W-MON"
    )

    return generate_data(date_data)


@pytest.fixture(scope="module")
def toy_y(toy_X: pd.DataFrame) -> pd.Series:
    return pd.Series(data=rng.integers(low=0, high=100, size=toy_X.shape[0]), name="y")


@pytest.fixture(scope="module")
def mmm() -> MMM:
    return MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        control_columns=["control_1", "control_2"],
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
        yearly_seasonality=3,
        adstock_first=True,
        time_varying_intercept=False,
        time_varying_media=False,
    )


@pytest.mark.parametrize("idata_format", ["netcdf", "zarr"], indirect=True)
def test_autolog_mmm(idata_format, mmm, toy_X, toy_y) -> None:
    mlflow.set_experiment("pymc-marketing-test-suite-mmm")
    with mlflow.start_run() as run:
        draws = 10
        tune = 5
        chains = 1
        idata = mmm.fit(
            toy_X,
            toy_y,
            draws=draws,
            chains=chains,
            tune=tune,
            nuts_sampler="pymc",
        )

    assert mlflow.active_run() is None

    run_id = run.info.run_id
    inputs, params, metrics, tags, artifacts = get_run_data(run_id)

    param_checks(
        params=params,
        draws=draws,
        chains=chains,
        tune=tune,
        nuts_sampler="pymc",
    )

    assert params["adstock_name"] == "Geometric"
    assert params["saturation_name"] == "Logistic"

    metric_checks(metrics, "pymc")

    assert set(artifacts) == {
        "coords.json",
        _idata_artifact_name(idata_format),
        "model_graph.pdf",
        "model_repr.txt",
        "summary.html",
    }
    assert tags == {}

    assert len(inputs) == 1
    parsed_inputs = json.loads(inputs[0].dataset.profile)

    expected_features_shape = {
        "channel_data": [135, 2],
        "control_data": [135, 2],
        "dayofyear": [135],
    }

    # Handle both old and new scaling approaches
    if "target" in idata.constant_data:
        expected_features_shape["target"] = [135]
    elif "target_data" in idata.constant_data:
        expected_features_shape["target_data"] = [135]
        expected_features_shape["target_scale"] = []

    # Include channel scaling variables if present (new scaling approach)
    if "channel_scale" in idata.constant_data:
        expected_features_shape["channel_scale"] = [2]

    assert parsed_inputs["features_shape"] == expected_features_shape
    assert parsed_inputs["targets_shape"] == {
        "y": [135],
    }


def test_log_mmm_configuration_keeps_large_lift_metadata_as_artifact(mmm, mocker):
    calibration = json.dumps([{"data": "x" * 10_000}])
    mocker.patch.object(
        mmm,
        "create_idata_attrs",
        return_value={"model_config": "{}", "lift_test_calibrations": calibration},
    )
    log_params = mocker.patch.object(pmm_mlflow.mlflow, "log_params")
    mocker.patch.object(pmm_mlflow.mlflow, "log_param")
    log_text = mocker.patch.object(pmm_mlflow.mlflow, "log_text")

    pmm_mlflow.log_mmm_configuration(mmm)

    log_params.assert_called_once_with({"model_config": "{}"})
    log_text.assert_called_once_with(calibration, "lift_test_calibrations.json")


@pytest.fixture(scope="module")
def multidimensional_mmm() -> MMM:
    return MMM(
        date_column="date",
        channel_columns=["channel_1", "channel_2"],
        target_column="y",
        adstock=GeometricAdstock(l_max=4),
        saturation=LogisticSaturation(),
    )


@pytest.fixture(scope="module")
def toy_multidim_X() -> pd.DataFrame:
    # Simple data for multidimensional MMM test
    n_obs = 20
    date_data = pd.DataFrame(
        {
            "date": pd.date_range(start="2020-01-01", periods=n_obs, freq="W-MON"),
            "channel_1": rng.integers(low=0, high=100, size=n_obs),
            "channel_2": rng.integers(low=0, high=100, size=n_obs),
        }
    )
    return date_data


@pytest.fixture(scope="module")
def toy_multidim_y(toy_multidim_X: pd.DataFrame) -> pd.Series:
    return pd.Series(
        data=rng.integers(low=0, high=100, size=toy_multidim_X.shape[0]), name="y"
    )


@pytest.mark.parametrize("idata_format", ["netcdf", "zarr"], indirect=True)
def test_autolog_multidimensional_mmm(
    idata_format, multidimensional_mmm, toy_multidim_X, toy_multidim_y
) -> None:
    mlflow.set_experiment("pymc-marketing-test-suite-multidimensional-mmm")
    with mlflow.start_run() as run:
        draws = 10
        tune = 5
        chains = 1
        multidimensional_mmm.fit(
            toy_multidim_X,
            toy_multidim_y,
            draws=draws,
            chains=chains,
            tune=tune,
            nuts_sampler="pymc",
        )

    assert mlflow.active_run() is None

    run_id = run.info.run_id
    inputs, params, metrics, tags, artifacts = get_run_data(run_id)

    param_checks(
        params=params,
        draws=draws,
        chains=chains,
        tune=tune,
        nuts_sampler="pymc",
    )

    assert params["adstock_name"] == "Geometric"
    assert params["saturation_name"] == "Logistic"

    metric_checks(metrics, "pymc")

    assert set(artifacts) == {
        "coords.json",
        _idata_artifact_name(idata_format),
        "model_graph.pdf",
        "model_repr.txt",
        "summary.html",
    }
    assert tags == {}

    assert len(inputs) == 1


@pytest.fixture(scope="function")
def mock_idata() -> xr.DataTree:
    chains = 4
    draws = 100
    coords = {
        "chain": np.arange(chains),
        "draw": np.arange(draws),
    }
    posterior = xr.Dataset(
        data_vars={
            "mu": (("chain", "draw"), rng.random(size=(chains, draws))),
            "sigma": (("chain", "draw"), rng.random(size=(chains, draws))),
        },
        coords=coords,
    )
    sample_stats = xr.Dataset(
        data_vars={
            "diverging": (
                ("chain", "draw"),
                rng.integers(0, 2, size=(chains, draws)),
            ),
            "energy": (("chain", "draw"), rng.random(size=(chains, draws))),
        },
        coords=coords,
    )
    return xr.DataTree.from_dict(
        {"/posterior": posterior, "/sample_stats": sample_stats},
    )


@pytest.mark.parametrize("selected_group", ["posterior", "sample_stats"])
def test_log_sample_diagnostics_missing_group(mock_idata, selected_group: str) -> None:
    idata = xr.DataTree.from_dict({f"/{selected_group}": mock_idata[selected_group]})
    missing_group = "sample_stats" if selected_group == "posterior" else "posterior"
    match = rf"DataTree object does not contain the group {missing_group}."
    with pytest.raises(KeyError, match=match):
        log_sample_diagnostics(idata)


def test_force_load_idata_groups_visits_every_group(mock_idata, monkeypatch) -> None:
    """Regression test: ``_force_load_idata_groups`` must load every group.

    ``DataTree.groups`` yields ``/``-prefixed paths (e.g. ``"/posterior"``),
    so the previous ``hasattr(idata, group)`` guard was always ``False`` and
    the function silently loaded nothing. Spy on ``Dataset.load`` to assert
    each group's dataset is actually materialized.
    """
    loaded_vars: list[set[str]] = []
    original_load = xr.Dataset.load

    def spy_load(self, *args, **kwargs):
        loaded_vars.append(set(self.data_vars))
        return original_load(self, *args, **kwargs)

    monkeypatch.setattr(xr.Dataset, "load", spy_load)

    pmm_mlflow._force_load_idata_groups(mock_idata)

    all_loaded = set().union(*loaded_vars) if loaded_vars else set()
    assert {"mu", "sigma"} <= all_loaded
    assert {"diverging", "energy"} <= all_loaded


@pytest.fixture
def small_idata() -> xr.DataTree:
    posterior = xr.Dataset(
        {"mu": (("chain", "draw"), rng.normal(size=(2, 5)))},
        coords={"chain": [0, 1], "draw": np.arange(5)},
    )
    tree = xr.DataTree.from_dict({"/posterior": posterior})
    tree.attrs["model_id"] = "abc123"
    return tree


@pytest.mark.parametrize("artifact", ["idata.nc", "idata.zarr"], ids=["netcdf", "zarr"])
def test_log_inference_data_round_trips_through_mlflow(
    small_idata, tmp_path, artifact
) -> None:
    # A stale store at the target path must not break the write.
    (tmp_path / "idata.zarr").mkdir()

    with mlflow.start_run() as run:
        log_inference_data(small_idata, save_file=tmp_path / artifact)

    # Only the stale store is left behind.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["idata.zarr"]
    downloaded = mlflow.artifacts.download_artifacts(
        run_id=run.info.run_id,
        artifact_path=artifact,
        dst_path=str(tmp_path / "download"),
    )
    xr.testing.assert_identical(load_idata(downloaded).load(), small_idata)


def test_log_inference_data_requires_existing_parent(small_idata, tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="Parent directory"):
        log_inference_data(small_idata, save_file=tmp_path / "missing" / "idata.nc")


def test_log_inference_data_uses_file_name_as_artifact_name(
    small_idata, tmp_path
) -> None:
    (tmp_path / "sub").mkdir()

    with mlflow.start_run() as run:
        log_inference_data(small_idata, save_file=tmp_path / "sub" / "idata.nc")

    assert [f.path for f in MlflowClient().list_artifacts(run.info.run_id)] == [
        "idata.nc"
    ]
    assert list((tmp_path / "sub").iterdir()) == []


@pytest.mark.parametrize(
    ("artifact", "locked"),
    [("idata.nc", "idata.nc"), ("idata.zarr", "zarr.json")],
    ids=["netcdf", "zarr"],
)
def test_log_inference_data_tolerates_an_undeletable_temp_file(
    small_idata, tmp_path, monkeypatch, artifact, locked
) -> None:
    # Windows can keep a just-written file open past the upload; the fit must
    # not fail over a leftover temp dir once the artifact is logged.
    real_unlink = os.unlink

    def unlink(path, *args, **kwargs):
        if os.path.basename(str(path)) == locked:
            raise PermissionError(f"simulated lock on {path}")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", unlink)

    with mlflow.start_run() as run:
        log_inference_data(small_idata, save_file=tmp_path / artifact)

    assert [f.path for f in MlflowClient().list_artifacts(run.info.run_id)] == [
        artifact
    ]
    (leftover,) = tmp_path.iterdir()
    assert leftover.name.startswith(".pymc_marketing_idata_")


def test_log_versions_logs_only_installed_idata_backends(monkeypatch) -> None:
    real_version = metadata.version

    def version_without_zarr(package: str) -> str:
        if package == "zarr":
            raise metadata.PackageNotFoundError(package)
        if package == "netCDF4":
            return "1.7.2"
        return real_version(package)

    monkeypatch.setattr(pmm_mlflow.metadata, "version", version_without_zarr)

    with mlflow.start_run() as run:
        log_versions()

    params = get_run_data(run.info.run_id).params
    assert params["xarray_version"] == xr.__version__
    assert params["netcdf4_version"] == "1.7.2"
    assert params["h5netcdf_version"] == real_version("h5netcdf")
    assert "zarr_version" not in params


@pytest.mark.parametrize(
    ("logged", "expected"),
    [
        (["idata.nc"], "idata.nc"),
        (["idata.zarr"], "idata.zarr"),
        (["summary.html"], FileNotFoundError),
        (["idata.nc", "idata.zarr"], ValueError),
    ],
    ids=["netcdf", "zarr", "none", "both"],
)
def test_find_idata_artifact(tmp_path, logged, expected) -> None:
    with mlflow.start_run() as run:
        for name in logged:
            local = tmp_path / name
            if name.endswith(".zarr"):
                local.mkdir()
                (local / "zarr.json").write_text("{}")
            else:
                local.write_text("x")
            mlflow.log_artifact(str(local))

    if isinstance(expected, str):
        assert pmm_mlflow._find_idata_artifact(run.info.run_id) == expected
    else:
        with pytest.raises(expected, match=run.info.run_id):
            pmm_mlflow._find_idata_artifact(run.info.run_id)


def test_autolog_rejects_unknown_idata_format() -> None:
    before = (pm.sample, pm.find_MAP, MMM.fit)

    with pytest.raises(ValueError, match="idata_format"):
        autolog(idata_format="parquet")

    assert (pm.sample, pm.find_MAP, MMM.fit) == before


def test_autolog_zarr_requires_zarr(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "zarr", None)
    with pytest.raises(ImportError, match=r"pymc-marketing\[zarr\]"):
        autolog(idata_format="zarr")


def test_autolog_zarr_requires_a_writing_xarray(monkeypatch) -> None:
    monkeypatch.setattr(xr, "__version__", "2025.4.0")
    with pytest.raises(ImportError, match=r"xarray>=2025\.6\.0"):
        autolog(idata_format="zarr")


def test_autolog_disable_skips_the_zarr_preflight(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "zarr", None)
    autolog(idata_format="zarr", disable=True)


@pytest.mark.parametrize(
    ("bad_call", "error"),
    [
        ({"idata_format": "bogus"}, ValueError),
        ({"idata_format": "zarr"}, ImportError),
    ],
    ids=["unknown-format", "old-xarray"],
)
def test_rejected_autolog_call_keeps_autologging_disabled(
    mmm, toy_X, toy_y, monkeypatch, bad_call, error
) -> None:
    autolog(disable=True)

    # Only the rejected call sees the old xarray: the fit below reads data
    # through pandas, which itself refuses an xarray that old.
    with monkeypatch.context() as patch:
        patch.setattr(xr, "__version__", "2025.4.0")
        with pytest.raises(error):
            autolog(**bad_call)

    with mlflow.start_run() as run:
        mmm.fit(toy_X, toy_y, draws=2, chains=1, tune=1)
    assert get_run_data(run.info.run_id).artifacts == []


def test_autolog_called_twice_logs_the_last_format_once(mmm, toy_X, toy_y) -> None:
    autolog(idata_format="zarr")  # on top of the fixture's netcdf call

    with mlflow.start_run() as run:
        mmm.fit(toy_X, toy_y, draws=2, chains=1, tune=1)

    artifacts = get_run_data(run.info.run_id).artifacts
    assert "idata.zarr" in artifacts
    assert "idata.nc" not in artifacts


def test_autolog_does_not_stack_wrappers() -> None:
    patched = [pm.sample, pm.find_MAP, MMM.fit, CLVModel.fit, BassModel.fit]
    depths = [_wrapper_depth(f) for f in patched]

    autolog()
    autolog(log_mmm=False, log_clv=False, log_bass=False)

    assert [_wrapper_depth(f) for f in (pm.sample, pm.find_MAP)] == depths[:2]
    assert [_wrapper_depth(f) for f in (MMM.fit, CLVModel.fit, BassModel.fit)] == [
        0,
        0,
        0,
    ]


def test_autolog_keeps_wrappers_it_did_not_install() -> None:
    original = pmm_mlflow._without_autolog_wrappers(pm.find_MAP)

    @functools.wraps(original)
    def foreign(*args, **kwargs):
        return original(*args, **kwargs)

    pm.find_MAP = foreign
    try:
        autolog()

        assert pm.find_MAP.__wrapped__ is foreign
        assert pmm_mlflow._without_autolog_wrappers(pm.find_MAP) is foreign
    finally:
        pm.find_MAP = original


def test_autolog_disable_switches_the_patches_off(mmm, toy_X, toy_y) -> None:
    autolog(disable=True)

    with mlflow.start_run() as run:
        mmm.fit(toy_X, toy_y, draws=2, chains=1, tune=1)

    _, params, _, _, artifacts = get_run_data(run.info.run_id)
    assert artifacts == []
    assert params == {}


@pytest.mark.parametrize(
    ("owner", "name"),
    [(pm, "find_MAP"), (CLVModel, "fit"), (BassModel, "fit")],
    ids=["find_MAP", "clv-fit", "bass-fit"],
)
def test_autolog_disable_passes_straight_through(monkeypatch, owner, name) -> None:
    calls = []

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return "result"

    monkeypatch.setattr(owner, name, original)
    autolog()  # wraps `original`
    autolog(disable=True)

    with mlflow.start_run() as run:
        assert getattr(owner, name)(object()) == "result"

    assert len(calls) == 1
    _, params, _, _, artifacts = get_run_data(run.info.run_id)
    assert params == {}
    assert artifacts == []


def test_autolog_disable_skips_error_logging(bad_starting_point_model) -> None:
    autolog(disable=True)

    with mlflow.start_run() as run, pytest.raises(SamplingError):
        pm.sample(
            draws=5,
            tune=5,
            chains=1,
            model=bad_starting_point_model,
            nuts_sampler="pymc",
        )

    assert get_run_data(run.info.run_id).artifacts == []


@pytest.fixture
def fitted_mmm_run(idata_format, mmm, toy_X, toy_y) -> tuple[MMM, str]:
    with mlflow.start_run() as run:
        mmm.fit(toy_X, toy_y, draws=2, chains=1, tune=1)
    return mmm, run.info.run_id


@pytest.mark.parametrize("idata_format", ["netcdf", "zarr"], indirect=True)
@pytest.mark.parametrize("keep_idata", [False, True], ids=["drop", "keep"])
def test_load_mmm_round_trips_and_honours_keep_idata(
    fitted_mmm_run, tmp_path, monkeypatch, keep_idata
) -> None:
    mmm, run_id = fitted_mmm_run
    monkeypatch.chdir(tmp_path)

    loaded = load_mmm(run_id, full_model=True, keep_idata=keep_idata)

    assert (tmp_path / f"idata_{run_id}").exists() is keep_idata
    assert loaded.idata.attrs["mlflow_run_id"] == run_id
    xr.testing.assert_equal(
        loaded.idata["posterior"].to_dataset(), mmm.idata["posterior"].to_dataset()
    )


@pytest.mark.parametrize("idata_format", ["netcdf", "zarr"], indirect=True)
def test_load_mmm_keeps_other_files_in_user_dst_path(
    fitted_mmm_run, tmp_path, recwarn
) -> None:
    _, run_id = fitted_mmm_run
    dst = tmp_path / "downloads"
    dst.mkdir()
    (dst / "notes.txt").write_text("mine")

    load_mmm(run_id, full_model=True, keep_idata=False, dst_path=str(dst))

    assert [p.name for p in dst.iterdir()] == ["notes.txt"]
    assert not [w for w in recwarn if "Could not remove" in str(w.message)]


@pytest.mark.parametrize("idata_format", ["zarr"], indirect=True)
def test_load_mmm_replaces_a_stale_zarr_download(fitted_mmm_run, tmp_path) -> None:
    _, run_id = fitted_mmm_run
    stale = tmp_path / "downloads" / "idata.zarr"
    stale.mkdir(parents=True)
    (stale / "leftover.txt").write_text("from an earlier download")

    loaded = load_mmm(
        run_id, full_model=True, keep_idata=True, dst_path=str(tmp_path / "downloads")
    )

    assert "posterior" in loaded.idata.children
    assert not (stale / "leftover.txt").exists()


@pytest.mark.parametrize("idata_format", ["netcdf", "zarr"], indirect=True)
def test_load_mmm_from_run_accepts_a_custom_artifact_name(
    idata_format, mmm, toy_X, toy_y, tmp_path
) -> None:
    autolog(disable=True)
    artifact = _idata_artifact_name(idata_format).replace("idata", "custom")
    with mlflow.start_run() as run:
        mmm.fit(toy_X, toy_y, draws=2, chains=1, tune=1)
        log_inference_data(mmm.idata, save_file=tmp_path / artifact)
    dst = tmp_path / "downloads"
    dst.mkdir()

    loaded = load_mmm_from_run(
        run.info.run_id, dst_path=str(dst), keep_idata=True, artifact=artifact
    )

    assert [p.name for p in dst.iterdir()] == [artifact]
    xr.testing.assert_equal(
        loaded.idata["posterior"].to_dataset(), mmm.idata["posterior"].to_dataset()
    )


@pytest.mark.parametrize(
    "artifact",
    ["{outside}", "../sibling.nc", "", ".", "..", "idata.nc/", "sub/idata.nc"],
    ids=["absolute", "parent", "empty", "dot", "dotdot", "trailing-slash", "nested"],
)
def test_load_mmm_from_run_rejects_artifacts_that_are_not_plain_names(
    tmp_path, artifact
) -> None:
    dst = tmp_path / "downloads"
    dst.mkdir()
    (dst / "mine.txt").write_text("keep")
    outside = tmp_path / "sibling.nc"
    outside.write_text("keep")
    with mlflow.start_run() as run:
        pass

    with pytest.raises(ValueError, match="plain name at the root of the run"):
        load_mmm_from_run(
            run.info.run_id,
            dst_path=str(dst),
            artifact=artifact.format(outside=outside),
        )

    assert outside.read_text() == "keep"
    assert (dst / "mine.txt").read_text() == "keep"


def test_load_mmm_from_run_keeps_a_local_copy_when_the_run_lacks_the_artifact(
    tmp_path,
) -> None:
    dst = tmp_path / "downloads"
    dst.mkdir()
    (dst / "my_model.nc").write_text("my own copy")
    with mlflow.start_run() as run:
        pass

    with pytest.raises(FileNotFoundError, match=r"no artifact 'my_model\.nc'"):
        load_mmm_from_run(
            run.info.run_id, dst_path=str(dst), keep_idata=True, artifact="my_model.nc"
        )

    assert (dst / "my_model.nc").read_text() == "my own copy"


def test_load_mmm_without_idata_artifact_raises() -> None:
    with mlflow.start_run() as run:
        pass

    with pytest.raises(FileNotFoundError, match="no inference-data artifact"):
        load_mmm(run.info.run_id, full_model=True)


def test_load_mmm_default_cleanup_keeps_unrelated_files(
    fitted_mmm_run, tmp_path, monkeypatch
) -> None:
    """A pre-existing ``idata_{run_id}`` directory belongs to the user, not to this call."""
    _, run_id = fitted_mmm_run
    monkeypatch.chdir(tmp_path)
    user_dir = tmp_path / f"idata_{run_id}"
    user_dir.mkdir()
    (user_dir / "other.txt").write_text("mine")

    load_mmm(run_id, full_model=True)

    assert [p.name for p in user_dir.iterdir()] == ["other.txt"]
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".idata_")]


@pytest.mark.parametrize("idata_format", ["netcdf", "zarr"], indirect=True)
def test_load_mmm_removes_download_when_load_fails(
    fitted_mmm_run, tmp_path, monkeypatch
) -> None:
    _, run_id = fitted_mmm_run
    dst = tmp_path / "downloads"
    dst.mkdir()

    def broken_load(cls, fname, check=True):
        raise RuntimeError("boom")

    monkeypatch.setattr(MMM, "load", classmethod(broken_load))
    with pytest.raises(RuntimeError, match="boom"):
        load_mmm(run_id, full_model=True, keep_idata=False, dst_path=str(dst))

    assert list(dst.iterdir()) == []


@pytest.fixture
def clv_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": [0, 1, 2, 3],
            "frequency": [0, 1, 1, 3],
            "recency": [0, 2, 2, 3],
            "T": [1, 2, 5, 3],
        }
    )


@pytest.mark.parametrize("idata_format", ["netcdf", "zarr"], indirect=True)
@pytest.mark.parametrize("model_cls", [BetaGeoModel])
def test_clv_fit_mcmc(idata_format, model_cls, clv_data) -> None:
    mlflow.set_experiment("pymc-marketing-test-suite-clv")

    sampler_config = {
        "draws": 2,
        "chains": 1,
        "tune": 1,
    }

    model = model_cls(sampler_config=sampler_config)
    with mlflow.start_run() as run:
        model.fit(data=clv_data)

    assert mlflow.active_run() is None

    run_id = run.info.run_id
    inputs, params, metrics, tags, artifacts = get_run_data(run_id)

    assert isinstance(inputs, list)

    assert params["fit_method"] == "mcmc"

    divergence_metric = {"total_divergences", "sampling_time_divergences"} & set(
        metrics.keys()
    )
    assert len(divergence_metric) == 1, (
        f"Expected exactly one divergence metric, got {divergence_metric} from {set(metrics.keys())}"
    )

    assert tags == {}

    assert set(artifacts) == {
        "coords.json",
        "model_repr.txt",
        "model_graph.pdf",
        "summary.html",
        _idata_artifact_name(idata_format),
    }


@pytest.mark.parametrize("idata_format", ["netcdf", "zarr"], indirect=True)
@pytest.mark.parametrize("model_cls", [BetaGeoModel])
def test_clv_fit_map(idata_format, model_cls, clv_data) -> None:
    mlflow.set_experiment("pymc-marketing-test-suite-clv")

    model = model_cls()
    with mlflow.start_run() as run:
        model.fit(data=clv_data, method="map")

    assert mlflow.active_run() is None

    run_id = run.info.run_id
    inputs, params, metrics, tags, artifacts = get_run_data(run_id)

    assert inputs == []

    assert params["fit_method"] == "map"

    assert set(metrics.keys()) == set()

    assert tags == {}

    assert set(artifacts) == {
        "coords.json",
        "model_repr.txt",
        "model_graph.pdf",
        _idata_artifact_name(idata_format),
    }


@pytest.fixture
def bass_data() -> np.ndarray:
    return np.random.default_rng(42).poisson(lam=100, size=20)


@pytest.mark.parametrize(
    "idata_format",
    [
        "netcdf",
        pytest.param(
            "zarr",
            marks=pytest.mark.xfail(
                CASE_INSENSITIVE_FS,
                reason="BassModel constant_data holds both T and t (#3177)",
                strict=True,
            ),
        ),
    ],
    indirect=True,
)
def test_autolog_bass(idata_format, bass_data) -> None:
    mlflow.set_experiment("pymc-marketing-test-suite-bass")

    sampler_config = {
        "draws": 2,
        "chains": 1,
        "tune": 1,
        # Force the pymc sampler so sampling_time / time_per_draw metrics are
        # populated; the pymc6 default sampler (nutpie) does not log them.
        "nuts_sampler": "pymc",
    }
    # Positive prior on m keeps the Poisson rate valid when the model graph
    # is rendered (it draws from the prior to evaluate shapes), so
    # model_graph.pdf is logged deterministically
    model_config = {
        "m": Prior("Normal", mu=100, sigma=10),
    }

    model = BassModel(model_config=model_config, sampler_config=sampler_config)
    with mlflow.start_run() as run:
        idata = model.fit(data=bass_data, random_seed=42)

    assert mlflow.active_run() is None
    assert idata.attrs["mlflow_run_id"] == run.info.run_id

    run_id = run.info.run_id
    inputs, params, metrics, tags, artifacts = get_run_data(run_id)

    assert isinstance(inputs, list)

    assert params["model_type"] == "BassModel"
    assert params["version"] == __version__

    # The Bass model builds its variables with pymc.dims, which wraps every RV
    # in a generic XRV; the logged name must still be the distribution.
    assert params["likelihood"] == "Poisson"

    model_config_logged = json.loads(params["model_config"])
    assert set(model_config_logged.keys()) == {"m", "p", "q", "likelihood"}

    sampler_config_logged = json.loads(params["sampler_config"])
    assert sampler_config_logged["draws"] == 2

    assert set(metrics.keys()) == {
        "total_divergences",
        "sampling_time",
        "time_per_draw",
    }

    assert tags == {}

    assert set(artifacts) == {
        "coords.json",
        "model_repr.txt",
        "model_graph.pdf",
        "summary.html",
        _idata_artifact_name(idata_format),
    }


@pytest.fixture(scope="function")
def mock_idata_for_loo() -> xr.DataTree:
    chains = 2
    draws = 50
    obs = 10
    coords = {
        "chain": np.arange(chains),
        "draw": np.arange(draws),
        "obs_id": np.arange(obs),
    }

    # Create log likelihood values for testing
    log_likelihood = xr.Dataset(
        data_vars={
            "obs": (("chain", "draw", "obs_id"), rng.normal(size=(chains, draws, obs))),
        },
        coords=coords,
    )

    posterior = xr.Dataset(
        data_vars={
            "mu": (("chain", "draw"), rng.random(size=(chains, draws))),
            "sigma": (("chain", "draw"), rng.random(size=(chains, draws))),
        },
        coords=coords,
    )

    sample_stats = xr.Dataset(
        data_vars={
            "diverging": (
                ("chain", "draw"),
                rng.integers(0, 2, size=(chains, draws)),
            ),
            "energy": (("chain", "draw"), rng.random(size=(chains, draws))),
        },
        coords=coords,
    )

    return xr.DataTree.from_dict(
        {
            "/posterior": posterior,
            "/sample_stats": sample_stats,
            "/log_likelihood": log_likelihood,
        },
    )


def test_log_mmm_evaluation_metrics() -> None:
    """Test logging of summary metrics to MLflow."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([[1.1, 2.1, 3.1]]).T
    custom_metrics = ["r_squared", "rmse"]

    prefix: str = "in-sample"
    with mlflow.start_run() as run:
        log_mmm_evaluation_metrics(
            y_true,
            y_pred,
            metrics_to_calculate=custom_metrics,
            hdi_prob=0.94,
            prefix=prefix,
        )

    run_id = run.info.run_id
    run_data = get_run_data(run_id)

    # Check that metrics are logged with expected prefixes and suffixes
    metric_prefixes = {"r_squared", "rmse"}
    metric_suffixes = {
        "mean",
        "median",
        "std",
        "min",
        "max",
        "94_hdi_lower",
        "94_hdi_upper",
    }
    expected_metrics = {
        f"{prefix}_{metric_prefix}_{metrix_suffix}"
        for metric_prefix in metric_prefixes
        for metrix_suffix in metric_suffixes
    }
    assert set(run_data.metrics.keys()) == expected_metrics

    assert all(isinstance(value, float) for value in run_data.metrics.values())


def test_callback_raises() -> None:
    match = r"At least one of"
    with pytest.raises(ValueError, match=match):
        create_log_callback()


def test_logging_callback(model_with_likelihood) -> None:
    mlflow.set_experiment("pymc-marketing-test-suite-logging-callback")

    callback = create_log_callback(
        stats=["energy"],
        parameters=["mu"],
        take_every=10,
    )
    with mlflow.start_run() as run:
        pm.sample(
            model=model_with_likelihood,
            draws=100,
            tune=1,
            chains=2,
            callback=callback,
        )

    assert mlflow.active_run() is None

    run_id = run.info.run_id
    client = MlflowClient()

    for chain in [0, 1]:
        for value in ["energy", "mu"]:
            history = client.get_metric_history(run_id, f"chain_{chain}/{value}")
            assert len(history) == 10


def _metric_history(run_id: str, key: str) -> tuple[np.ndarray, np.ndarray]:
    history = sorted(
        MlflowClient().get_metric_history(run_id, key), key=lambda m: m.step
    )
    return np.array([m.step for m in history]), np.array([m.value for m in history])


def test_logging_callback_logs_constrained_values(model_with_likelihood) -> None:
    # `sigma` is a HalfNormal, so it is sampled as `sigma_log__`. The metric
    # named `sigma` must hold sigma itself, not its log.
    mlflow.set_experiment("pymc-marketing-test-suite-log-constrained-values")

    tune = 100
    callback = create_log_callback(parameters=["mu", "sigma"], take_every=10)
    with mlflow.start_run() as run:
        idata = pm.sample(
            model=model_with_likelihood,
            draws=100,
            tune=tune,
            chains=1,
            callback=callback,
        )

    for name in ["mu", "sigma"]:
        steps, values = _metric_history(run.info.run_id, f"chain_0/{name}")
        assert len(values) == 10
        assert np.unique(values).size > 1
        expected = idata.posterior[name].sel(chain=0).values[steps - tune]
        np.testing.assert_allclose(values, expected)


def test_logging_callback_logs_other_transforms_and_deterministics() -> None:
    mlflow.set_experiment("pymc-marketing-test-suite-log-other-transforms")

    with pm.Model() as model:
        p = pm.Beta("p", alpha=2, beta=2)  # sampled as `p_logodds__`
        b = pm.Uniform("b", lower=-3, upper=7)  # sampled as `b_interval__`
        pm.Deterministic("scaled_p", 10 * p)
        pm.Normal("obs", mu=b * p, sigma=1, observed=rng.normal(size=10))

    tune = 100
    parameters = ["p", "b", "scaled_p"]
    callback = create_log_callback(parameters=parameters, take_every=10)
    with mlflow.start_run() as run:
        idata = pm.sample(model=model, draws=50, tune=tune, chains=1, callback=callback)

    for name in parameters:
        steps, values = _metric_history(run.info.run_id, f"chain_0/{name}")
        assert len(values) == 5
        assert np.unique(values).size > 1
        expected = idata.posterior[name].sel(chain=0).values[steps - tune]
        np.testing.assert_allclose(values, expected)


def test_logging_callback_explicit_transformed_name(model_with_likelihood) -> None:
    # Passing the sampler's value var name logs the unconstrained value.
    mlflow.set_experiment("pymc-marketing-test-suite-log-transformed-name")

    tune = 100
    callback = create_log_callback(parameters=["sigma_log__"], take_every=10)
    with mlflow.start_run() as run:
        idata = pm.sample(
            model=model_with_likelihood,
            draws=100,
            tune=tune,
            chains=1,
            callback=callback,
        )

    steps, values = _metric_history(run.info.run_id, "chain_0/sigma_log__")
    assert len(values) == 10
    assert np.unique(values).size > 1
    expected = np.log(idata.posterior["sigma"].sel(chain=0).values[steps - tune])
    np.testing.assert_allclose(values, expected)


class _TraceWithoutRecordedDraws(IBaseTrace):
    """Like ``ZarrChain``, keeps the unimplemented ``__len__`` and ``point``."""


def test_logging_callback_falls_back_to_draw_point(mocker) -> None:
    log_metric = mocker.patch.object(pmm_mlflow.mlflow, "log_metric")
    callback = create_log_callback(parameters=["mu", "sigma_log__"], take_every=1)
    draw = SimpleNamespace(
        chain=0,
        draw_idx=3,
        tuning=False,
        stats=[{}],
        point={"mu": 0.5, "sigma_log__": -1.0},
    )

    callback(_TraceWithoutRecordedDraws(), draw)

    log_metric.assert_has_calls(
        [
            mocker.call(key="chain_0/mu", value=0.5, step=3),
            mocker.call(key="chain_0/sigma_log__", value=-1.0, step=3),
        ]
    )


def test_logging_callback_stats_only_does_not_read_trace(mocker) -> None:
    log_metric = mocker.patch.object(pmm_mlflow.mlflow, "log_metric")
    callback = create_log_callback(stats=["energy"], take_every=1)
    trace = mocker.MagicMock()
    draw = SimpleNamespace(
        chain=0,
        draw_idx=2,
        tuning=False,
        stats=[{"energy": 1.5}],
        point={"mu": 0.0},
    )

    callback(trace, draw)

    log_metric.assert_called_once_with(key="chain_0/energy", value=1.5, step=2)
    trace.point.assert_not_called()


def test_logging_callback_unknown_parameter_raises() -> None:
    callback = create_log_callback(parameters=["nope"], take_every=1)
    draw = SimpleNamespace(
        chain=0,
        draw_idx=0,
        tuning=False,
        stats=[{}],
        point={"mu": 0.0, "sigma_log__": 0.0},
    )

    with pytest.raises(KeyError, match=r"'nope' not found in the recorded draw"):
        callback(_TraceWithoutRecordedDraws(), draw)


def test_log_error() -> None:
    mlflow.set_experiment("pymc-marketing-test-suite-log-error")

    class MyException(Exception):
        """Custom exception for testing purposes."""

    def foo():
        raise MyException("This is an error")

    def bar():
        foo()

    def baz():
        bar()

    file_name = "sample-error.txt"
    main = log_error(baz, file_name=file_name)

    with mlflow.start_run() as run:
        with pytest.raises(MyException, match=r"This is an error"):
            main()

    assert mlflow.active_run() is None

    run_data = get_run_data(run.info.run_id)

    assert run_data.artifacts == [file_name]

    artifact_uri = f"{run.info.artifact_uri}/{file_name}"
    loaded_artifact = mlflow.artifacts.load_text(artifact_uri)

    lines = [
        "in baz",
        "in bar",
        "in foo",
        "This is an error",
    ]
    for line in lines:
        assert line in loaded_artifact


@pytest.mark.parametrize(
    "doc_source",
    [pmm_mlflow.__doc__, log_mmm.__doc__, autolog.__doc__],
    ids=["module", "log_mmm", "autolog"],
)
def test_mlflow_docstrings_have_no_removed_methods(doc_source):
    """Guard against re-introducing removed MMM API methods in published examples.

    The old ``MMM.plot_components_contributions`` helper no longer exists on
    the new ``mmm.MMM``; copy-pasting the obsolete docstring
    example would raise ``AttributeError`` at runtime, so make sure none of
    the rendered Sphinx docstrings still reference it.
    """
    assert doc_source is not None
    assert "plot_components_contributions" not in doc_source
