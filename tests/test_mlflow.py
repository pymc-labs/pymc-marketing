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
import json
import logging
import threading
from collections import namedtuple
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
from pymc_marketing.mlflow import (
    autolog,
    create_log_callback,
    create_nutpie_log_callback,
    log_error,
    log_likelihood_type,
    log_mmm,
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

    # Restore to original (unwrap any wrappers applied during this test)
    while hasattr(pm.sample, "__wrapped__"):
        pm.sample = pm.sample.__wrapped__
    while hasattr(MMM.fit, "__wrapped__"):
        MMM.fit = MMM.fit.__wrapped__
    while hasattr(BassModel.fit, "__wrapped__"):
        BassModel.fit = BassModel.fit.__wrapped__


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
    # numpyro and blackjax do not report sampling time; nutpie does, on
    # `posterior.attrs` rather than `sample_stats.attrs`.
    if nuts_sampler not in ["numpyro", "blackjax"]:
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


def test_autolog_multidimensional_mmm(
    multidimensional_mmm, toy_multidim_X, toy_multidim_y
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
        "idata.nc",
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
        "idata.nc",
    }


@pytest.mark.parametrize("model_cls", [BetaGeoModel])
def test_clv_fit_map(model_cls, clv_data) -> None:
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
        "idata.nc",
    }


@pytest.fixture
def bass_data() -> np.ndarray:
    return np.random.default_rng(42).poisson(lam=100, size=20)


def test_autolog_bass(bass_data) -> None:
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
        "idata.nc",
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


def _draw(chain=0, draw_idx=0, tuning=False, stats=None, point=None):
    return SimpleNamespace(
        chain=chain,
        draw_idx=draw_idx,
        tuning=tuning,
        stats=[stats or {}],
        point=point or {},
    )


def test_logging_callback_logs_cumulative_divergences(mocker) -> None:
    """`divergences` is passed through as-is: pymc already counts them."""
    log_metric = mocker.patch.object(pmm_mlflow.mlflow, "log_metric")
    callback = create_log_callback(stats=["divergences"], take_every=2)
    trace = mocker.MagicMock()

    for idx, divergences in enumerate([0.0, 0.0, 1.0, 2.0, 2.0, 3.0]):
        callback(trace, _draw(draw_idx=idx, stats={"divergences": divergences}))

    assert log_metric.call_args_list == [
        mocker.call(key="chain_0/divergences", value=0.0, step=0),
        mocker.call(key="chain_0/divergences", value=1.0, step=2),
        mocker.call(key="chain_0/divergences", value=2.0, step=4),
    ]


def test_logging_callback_take_every_bounds_the_writes(mocker) -> None:
    """The number of store writes follows `take_every`, not the draw count."""
    log_metric = mocker.patch.object(pmm_mlflow.mlflow, "log_metric")
    callback = create_log_callback(stats=["divergences", "step_size"], take_every=100)
    trace = mocker.MagicMock()

    for idx in range(1000):
        callback(
            trace,
            _draw(draw_idx=idx, stats={"divergences": 1.0, "step_size": 0.1}),
        )

    # 10 draws on the grid x 2 stats.
    assert log_metric.call_count == 20


def test_logging_callback_rebase_steps(mocker) -> None:
    """Post-tuning draws are logged starting at step 0."""
    log_metric = mocker.patch.object(pmm_mlflow.mlflow, "log_metric")
    callback = create_log_callback(
        stats=["energy"],
        rebase_steps=True,
        take_every=1,
    )
    trace = mocker.MagicMock()

    # Tuning draws are skipped, but still set each chain's offset.
    for idx in range(3):
        callback(trace, _draw(draw_idx=idx, tuning=True, stats={"energy": 9.0}))
    for idx in range(3, 6):
        callback(trace, _draw(draw_idx=idx, stats={"energy": 1.0}))

    assert log_metric.call_args_list == [
        mocker.call(key="chain_0/energy", value=1.0, step=0),
        mocker.call(key="chain_0/energy", value=1.0, step=1),
        mocker.call(key="chain_0/energy", value=1.0, step=2),
    ]


def test_logging_callback_step_not_rebased_by_default(mocker) -> None:
    """By default the raw draw_idx is kept, so the axis still counts tuning."""
    log_metric = mocker.patch.object(pmm_mlflow.mlflow, "log_metric")
    callback = create_log_callback(stats=["energy"], take_every=1)
    trace = mocker.MagicMock()

    for idx in range(3):
        callback(trace, _draw(draw_idx=idx, tuning=True, stats={"energy": 9.0}))
    callback(trace, _draw(draw_idx=3, stats={"energy": 1.0}))

    log_metric.assert_called_once_with(key="chain_0/energy", value=1.0, step=3)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"stats": ["energy"], "rebase_steps": True, "exclude_tuning": False},
            "requires `exclude_tuning=True`",
        ),
    ],
)
def test_logging_callback_invalid_arguments(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        create_log_callback(**kwargs)


class _Chain:
    """Stand-in for `nutpie.ChainProgress`."""

    def __init__(
        self,
        finished_draws=100,
        total_draws=1000,
        tuning=False,
        divergences=0,
        step_size=0.1,
        latest_num_steps=7,
        total_num_steps=70,
        runtime_ms=1500.0,
    ):
        self.finished_draws = finished_draws
        self.total_draws = total_draws
        self.tuning = tuning
        self.divergences = divergences
        self.step_size = step_size
        self.latest_num_steps = latest_num_steps
        self.total_num_steps = total_num_steps
        self.runtime_ms = runtime_ms


def test_nutpie_callback_logs_chain_progress(mocker) -> None:
    """Stats are logged per chain at `finished_draws`, keyed like the pymc path."""
    client = mocker.patch.object(pmm_mlflow.mlflow.tracking, "MlflowClient")
    create_nutpie_log_callback(stats=["divergences", "step_size"], run_id="run-1")
    log_metric = client.return_value.log_metric

    callback = create_nutpie_log_callback(
        stats=["divergences", "step_size"],
        run_id="run-1",
        min_interval=0.0,
    )
    callback([_Chain(finished_draws=300, divergences=4), _Chain(finished_draws=250)])

    assert log_metric.call_args_list == [
        mocker.call("run-1", "chain_0/divergences", 4.0, step=300),
        mocker.call("run-1", "chain_0/step_size", 0.1, step=300),
        mocker.call("run-1", "chain_1/divergences", 0.0, step=250),
        mocker.call("run-1", "chain_1/step_size", 0.1, step=250),
    ]


def test_nutpie_callback_skips_unchanged_values(mocker) -> None:
    """A poll that reports the same numbers does not write again."""
    client = mocker.patch.object(pmm_mlflow.mlflow.tracking, "MlflowClient")
    callback = create_nutpie_log_callback(
        stats=["divergences"], run_id="r", min_interval=0.0
    )

    callback([_Chain(divergences=2)])
    callback([_Chain(divergences=2)])

    assert client.return_value.log_metric.call_count == 1


def test_nutpie_callback_throttles_by_min_interval(mocker) -> None:
    """`min_interval` bounds the write rate; nutpie polls far more often."""
    client = mocker.patch.object(pmm_mlflow.mlflow.tracking, "MlflowClient")
    callback = create_nutpie_log_callback(
        stats=["divergences"], run_id="r", min_interval=60.0
    )

    for i in range(10):
        callback([_Chain(finished_draws=i, divergences=i)])

    assert client.return_value.log_metric.call_count == 1


def test_nutpie_callback_divergences_new_is_a_delta(mocker) -> None:
    """`divergences_new` marks only the polls where a burst happened."""
    client = mocker.patch.object(pmm_mlflow.mlflow.tracking, "MlflowClient")
    callback = create_nutpie_log_callback(
        stats=["divergences_new"],
        run_id="r",
        min_interval=0.0,
    )

    callback([_Chain(finished_draws=100, divergences=0)])
    callback([_Chain(finished_draws=200, divergences=5)])
    callback([_Chain(finished_draws=300, divergences=5)])

    calls = client.return_value.log_metric.call_args_list
    assert [c.args[2] for c in calls] == [5.0]
    assert calls[0].kwargs["step"] == 200


def test_nutpie_callback_excludes_tuning(mocker) -> None:
    """Chains still tuning are skipped by default."""
    client = mocker.patch.object(pmm_mlflow.mlflow.tracking, "MlflowClient")
    callback = create_nutpie_log_callback(
        stats=["divergences"], run_id="r", min_interval=0.0
    )

    callback([_Chain(tuning=True), _Chain(finished_draws=10)])

    keys = [c.args[1] for c in client.return_value.log_metric.call_args_list]
    assert keys == ["chain_1/divergences"]


def test_nutpie_callback_rebase_steps(mocker) -> None:
    """Post-tuning draws are logged from step 0."""
    client = mocker.patch.object(pmm_mlflow.mlflow.tracking, "MlflowClient")
    callback = create_nutpie_log_callback(
        stats=["divergences"],
        run_id="r",
        min_interval=0.0,
        rebase_steps=True,
    )

    callback([_Chain(finished_draws=1000, tuning=True)])
    callback([_Chain(finished_draws=1200)])
    callback([_Chain(finished_draws=1300, divergences=1)])

    steps = [c.kwargs["step"] for c in client.return_value.log_metric.call_args_list]
    assert steps == [0, 100]


def test_nutpie_callback_uses_active_run(mocker) -> None:
    """Without an explicit `run_id`, the active run is captured up front."""
    client = mocker.patch.object(pmm_mlflow.mlflow.tracking, "MlflowClient")
    active_run = SimpleNamespace(info=SimpleNamespace(run_id="active-run"))
    mocker.patch.object(pmm_mlflow.mlflow, "active_run", return_value=active_run)

    callback = create_nutpie_log_callback(stats=["divergences"], min_interval=0.0)
    callback([_Chain(divergences=1)])

    assert client.return_value.log_metric.call_args.args[0] == "active-run"


def test_nutpie_callback_requires_a_run(mocker) -> None:
    mocker.patch.object(pmm_mlflow.mlflow, "active_run", return_value=None)

    with pytest.raises(ValueError, match="No active run"):
        create_nutpie_log_callback(stats=["divergences"])


def test_nutpie_callback_rejects_unknown_stats() -> None:
    with pytest.raises(ValueError, match="Unknown stats"):
        create_nutpie_log_callback(stats=["not_a_stat"], run_id="r")


def test_nutpie_callback_swallows_and_logs_errors(mocker, caplog) -> None:
    """nutpie discards callback exceptions, so failures must be visible here."""
    client = mocker.patch.object(pmm_mlflow.mlflow.tracking, "MlflowClient")
    client.return_value.log_metric.side_effect = RuntimeError("store is down")
    callback = create_nutpie_log_callback(
        stats=["divergences"], run_id="r", min_interval=0.0
    )

    with caplog.at_level(logging.ERROR, logger="pymc_marketing.mlflow"):
        callback([_Chain(divergences=1)])

    assert "store is down" in caplog.text


def test_nutpie_callback_take_every_bounds_the_writes(mocker) -> None:
    """`take_every` caps writes by draws, independently of the poll rate."""
    client = mocker.patch.object(pmm_mlflow.mlflow.tracking, "MlflowClient")
    callback = create_nutpie_log_callback(
        stats=["step_size"],
        run_id="r",
        min_interval=0.0,
        take_every=100,
    )

    # 1000 draws polled every 10, with a value that changes every poll.
    for idx in range(0, 1000, 10):
        callback([_Chain(finished_draws=idx, step_size=idx / 1000)])

    assert client.return_value.log_metric.call_count == 10


def test_nutpie_callback_take_every_does_not_throttle_bursts(mocker) -> None:
    """`divergences_new` is logged even between `take_every` marks."""
    client = mocker.patch.object(pmm_mlflow.mlflow.tracking, "MlflowClient")
    callback = create_nutpie_log_callback(
        stats=["divergences_new"],
        run_id="r",
        min_interval=0.0,
        take_every=1000,
    )

    callback([_Chain(finished_draws=10, divergences=0)])
    callback([_Chain(finished_draws=20, divergences=3)])
    callback([_Chain(finished_draws=30, divergences=3)])

    calls = client.return_value.log_metric.call_args_list
    assert len(calls) == 1
    assert calls[0].args[2] == 3.0
    assert calls[0].kwargs["step"] == 20


def test_nutpie_callback_is_thread_safe(mocker) -> None:
    """nutpie calls the callback from a thread per chain; writes stay bounded."""
    client = mocker.patch.object(pmm_mlflow.mlflow.tracking, "MlflowClient")
    callback = create_nutpie_log_callback(
        stats=["divergences", "step_size"],
        run_id="r",
        min_interval=60.0,
    )

    # Four chains reporting at once, the way the sampler does.
    barrier = threading.Barrier(4)

    def poll(chain_id):
        barrier.wait()
        callback([_Chain(finished_draws=100 + chain_id, divergences=chain_id)])

    threads = [threading.Thread(target=poll, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # One round of writes only: at most 4 chains x 2 stats, not 4 rounds.
    assert client.return_value.log_metric.call_count <= 8


def test_nutpie_callback_against_a_real_sampler() -> None:
    """Cross-check the logged numbers against the trace nutpie returns.

    The callback reports what nuts-rs believes mid-run, the trace is written
    at the end, so agreeing on the divergence total is a real check on the
    progress contract rather than a restatement of the callback.
    """
    nutpie = pytest.importorskip("nutpie")

    # A funnel: cheap to sample, but it does diverge.
    model = pm.Model()
    with model:
        v = pm.Normal("v", 0, 3)
        pm.Normal("x", v, pm.math.exp(v / 2), shape=10)

    chains = 2
    client = mlflow.tracking.MlflowClient()

    with mlflow.start_run() as run:
        callback = create_nutpie_log_callback(
            stats=["divergences"],
            run_id=run.info.run_id,
            min_interval=0.0,
        )
        with model:
            compiled = nutpie.compile_pymc_model(model)
            idata = nutpie.sample(
                compiled,
                draws=200,
                tune=200,
                chains=chains,
                progress_bar=False,
                progress_callback=callback,
                progress_rate=1,
            )

    logged_totals = {}
    for chain_id in range(chains):
        history = client.get_metric_history(
            run.info.run_id, f"chain_{chain_id}/divergences"
        )
        values = [point.value for point in history]
        steps = [point.step for point in history]

        assert values, f"chain {chain_id} logged nothing"
        # Cumulative counters only ever go up, and steps never go back.
        assert values == sorted(values)
        assert steps == sorted(steps)
        logged_totals[chain_id] = values[-1]

    trace_total = int(idata.sample_stats["diverging"].sum())
    # The last poll can land just before the final draws, so allow a little
    # slack -- but it must never claim more divergences than actually happened.
    assert 0 <= trace_total - sum(logged_totals.values()) <= 2 * chains


def test_nutpie_callback_min_interval_starts_at_the_first_write(mocker) -> None:
    """A poll that only sees warmup must not spend the whole interval.

    Otherwise a long warmup swallows the budget and a run shorter than
    `min_interval` logs nothing at all.
    """
    client = mocker.patch.object(pmm_mlflow.mlflow.tracking, "MlflowClient")
    callback = create_nutpie_log_callback(
        stats=["divergences"], run_id="r", min_interval=60.0
    )

    callback([_Chain(tuning=True)])
    callback([_Chain(tuning=True, divergences=0)])
    callback([_Chain(finished_draws=1200)])

    assert client.return_value.log_metric.call_count == 1
