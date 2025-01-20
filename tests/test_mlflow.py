#   Copyright 2022 - 2025 The PyMC Labs Developers
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
import json
import logging
from collections import namedtuple
from pathlib import Path

import arviz as az
import mlflow
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr
from mlflow.client import MlflowClient

from pymc_marketing.clv import BetaGeoModel
from pymc_marketing.mlflow import (
    autolog,
    log_likelihood_type,
    log_mmm_evaluation_metrics,
    log_model_graph,
    log_sample_diagnostics,
)
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
from pymc_marketing.version import __version__

seed = sum(map(ord, "mlflow-with-pymc"))
rng = np.random.default_rng(seed)


@pytest.fixture(scope="function", autouse=True)
def setup_module():
    uri: str = "sqlite:///mlruns.db"
    mlflow.set_tracking_uri(uri=uri)
    autolog()

    yield

    pm.sample = pm.sample.__wrapped__
    MMM.fit = MMM.fit.__wrapped__


@pytest.fixture(scope="module")
def model() -> pm.Model:
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


def test_file_system_uri_supported(model) -> None:
    mlflow.set_tracking_uri(uri=Path("./mlruns"))
    mlflow.set_experiment("pymc-marketing-test-suite-local-file")
    with mlflow.start_run() as run:
        pm.sample(
            model=model,
            chains=1,
            tune=25,
            draws=30,
        )

    assert mlflow.get_tracking_uri().startswith("file:///")
    assert mlflow.active_run() is None

    run_id = run.info.run_id
    run_data = get_run_data(run_id)
    basic_logging_checks(run_data)


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


def test_multi_likelihood_type(multi_likelihood_model) -> None:
    mlflow.set_experiment("pymc-marketing-test-suite-multi-likelihood")
    with mlflow.start_run() as run:
        log_likelihood_type(multi_likelihood_model)

    run_id = run.info.run_id
    run_data = get_run_data(run_id)

    assert run_data.params == {
        "observed_RVs_types": "['Normal', 'Gamma']",
    }


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
    ],
    ids=["no_graphviz", "render_error"],
)
def test_log_model_graph_no_graphviz(
    caplog, mocker, model, to_patch, side_effect, expected_info_message
) -> None:
    mocker.patch(
        to_patch,
        side_effect=side_effect,
    )
    with mlflow.start_run() as run:
        with caplog.at_level(logging.INFO):
            log_model_graph(model, "model_graph")

    assert caplog.messages == [
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

    other_keys = ["pymc_version"]
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
def test_autolog_pymc_model(model, nuts_sampler) -> None:
    mlflow.set_experiment("pymc-marketing-test-suite-pymc-model")
    with mlflow.start_run() as run:
        draws = 30
        tune = 25
        chains = 2
        pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            model=model,
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
    return pd.Series(data=rng.integers(low=0, high=100, size=toy_X.shape[0]))


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


def test_autolog_mmm(mmm, toy_X, toy_y) -> None:
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

    assert params["adstock_name"] == "geometric"
    assert params["saturation_name"] == "logistic"

    metric_checks(metrics, "pymc")

    assert set(artifacts) == {
        "coords.json",
        "idata.nc",
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
    if "target" in idata.constant_data:
        expected_features_shape["target"] = [135]

    assert parsed_inputs["features_shape"] == expected_features_shape
    assert parsed_inputs["targets_shape"] == {
        "y": [135],
    }


@pytest.fixture(scope="function")
def mock_idata() -> az.InferenceData:
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
    return az.InferenceData(
        posterior=posterior,
        sample_stats=sample_stats,
    )


@pytest.mark.parametrize("selected_group", ["posterior", "sample_stats"])
def test_log_sample_diagnostics_missing_group(mock_idata, selected_group: str) -> None:
    idata = az.InferenceData(**{selected_group: mock_idata[selected_group]})
    missing_group = "sample_stats" if selected_group == "posterior" else "posterior"
    match = f"InferenceData object does not contain the group {missing_group}."
    with pytest.raises(KeyError, match=match):
        log_sample_diagnostics(idata)


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


@pytest.mark.parametrize("model_cls", [BetaGeoModel])
def test_clv_fit_mcmc(model_cls, clv_data) -> None:
    mlflow.set_experiment("pymc-marketing-test-suite-clv")

    sampler_config = {
        "draws": 2,
        "chains": 1,
        "tune": 1,
    }

    model = model_cls(data=clv_data, sampler_config=sampler_config)
    with mlflow.start_run() as run:
        model.fit()

    assert mlflow.active_run() is None

    run_id = run.info.run_id
    inputs, params, metrics, tags, artifacts = get_run_data(run_id)

    assert isinstance(inputs, list)

    assert params["fit_method"] == "mcmc"

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
        "idata.nc",
    }


@pytest.mark.parametrize("model_cls", [BetaGeoModel])
def test_clv_fit_map(model_cls, clv_data) -> None:
    mlflow.set_experiment("pymc-marketing-test-suite-clv")

    model = model_cls(data=clv_data)
    with mlflow.start_run() as run:
        model.fit(fit_method="map")

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
        "idata.nc",
    }


@pytest.fixture(scope="function")
def mock_idata_for_loo() -> az.InferenceData:
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

    return az.InferenceData(
        posterior=posterior,
        sample_stats=sample_stats,
        log_likelihood=log_likelihood,
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
