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
import pymc as pm
import pytest
from pymc.model_graph import ModelGraph

from pymc_marketing.model_graph import deterministics_to_flat


@pytest.fixture
def toy_model() -> pm.Model:
    coords = {"time": [1, 2, 3], "covariate": ["A", "B"]}
    with pm.Model(coords=coords) as model:
        x = pm.Normal("x", dims=("time", "covariate"))
        x_mean = pm.Deterministic("x_mean", x.mean(axis=0), dims="covariate")
        pm.Deterministic("centered", x - x_mean, dims=("time", "covariate"))

    return model


@pytest.fixture
def toy_model_given_x_mean(toy_model):
    return deterministics_to_flat(toy_model, ["x_mean"])


def test_original_model_is_not_modified(toy_model):
    assert toy_model.deterministics == [
        toy_model["x_mean"],
        toy_model["centered"],
    ]
    assert toy_model.free_RVs == [toy_model["x"]]
    model_graph = ModelGraph(toy_model)
    compute_graph = model_graph.make_compute_graph()
    assert compute_graph == {
        "x": set(),
        "x_mean": {"x"},
        "centered": {"x", "x_mean"},
    }


def get_rv_class_name(var):
    return var.owner.op.__class__.__name__


def test_is_flat_distribution(toy_model_given_x_mean):
    x_mean = toy_model_given_x_mean["x_mean"]
    assert get_rv_class_name(x_mean) == "FlatRV"


def test_no_inputs(toy_model_given_x_mean):
    model_graph = ModelGraph(toy_model_given_x_mean)
    compute_graph = model_graph.make_compute_graph()

    assert compute_graph == {
        "x": set(),
        "x_mean": set(),
        "centered": {"x", "x_mean"},
    }
