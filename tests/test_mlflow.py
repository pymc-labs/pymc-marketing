#   Copyright 2024 The PyMC Labs Developers
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

import mlflow
import numpy as np
import pandas as pd
import pymc as pm
import pytest
from mlflow.client import MlflowClient

from pymc_marketing.mlflow import autolog
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

uri: str = "sqlite:///mlruns.db"
mlflow.set_tracking_uri(uri=uri)

seed = sum(map(ord, "mlflow-with-pymc"))
rng = np.random.default_rng(seed)

autolog(end_run_after_sample=False)


def define_model() -> pm.Model:
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


def get_run_data(run_id):
    # Adapted from mlflow tests for sklearn autolog
    client = MlflowClient()
    run = client.get_run(run_id)
    data = run.data
    # Ignore tags mlflow logs by default (e.g. "mlflow.user")
    tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in client.list_artifacts(run_id)]
    inputs = [inp for inp in run.inputs.dataset_inputs]
    return inputs, data.params, data.metrics, tags, artifacts


def metric_checks(metrics, nuts_sampler) -> None:
    assert metrics["total_divergences"] >= 0.0
    if nuts_sampler not in ["numpyro", "nutpie", "blackjax"]:
        assert metrics["sampling_time"] >= 0.0
        assert metrics["time_per_draw"] >= 0.0


@pytest.mark.parametrize(
    "nuts_sampler",
    [
        "pymc",
        "numpyro",
        "nutpie",
        "blackjax",
    ],
)
def test_autolog_pymc_model(nuts_sampler) -> None:
    mlflow.set_experiment("pymc-marketing-test-suite-pymc-model")
    with mlflow.start_run() as run:
        model = define_model()

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

    assert params["draws"] == str(draws)
    assert params["chains"] == str(chains)
    if nuts_sampler not in ["numpyro", "blackjax"]:
        assert params["inference_library"] == nuts_sampler
    assert params["n_free_RVs"] == "2"
    assert params["n_observed_RVs"] == "1"
    assert params["n_deterministics"] == "0"
    assert params["n_potentials"] == "0"
    assert params["likelihood"] == "Normal"
    if nuts_sampler not in ["numpyro", "nutpie", "blackjax"]:
        assert params["tuning_steps"] == str(tune)

    other_keys = ["pymc_version"]
    if nuts_sampler not in ["numpyro", "blackjax"]:
        other_keys.extend(["inference_library_version"])

    for other_key in other_keys:
        assert other_key in params

    metric_checks(metrics, nuts_sampler)

    assert tags == {}
    assert artifacts == [
        "coords.json",
        "model_graph.pdf",
        "model_repr.txt",
        "summary.html",
    ]

    assert len(inputs) == 1
    mlflow.end_run()


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
        mmm.fit(toy_X, toy_y, draws=10, tune=5, chains=1)

    run_id = run.info.run_id
    inputs, params, metrics, tags, artifacts = get_run_data(run_id)

    assert params["adstock_name"] == "geometric"
    assert params["saturation_name"] == "logistic"

    metric_checks(metrics, "pymc")

    assert "idata.nc" in artifacts
    assert tags == {}
    assert len(inputs) == 1

    parsed_inputs = json.loads(inputs[0].dataset.profile)
    assert parsed_inputs["features_shape"] == {
        "channel_data": [135, 2],
        "control_data": [135, 2],
    }
    assert parsed_inputs["targets_shape"] == {
        "y": [135],
    }

    mlflow.end_run()
