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
"""Tests for model-implied causal DAGs on MMM."""

import builtins
from unittest.mock import patch

import graphviz
import networkx as nx
import numpy as np
import pytest
import xarray as xr
from pydantic import ConfigDict

from pymc_marketing.mmm.additive_effect import (
    ControlMuEffect,
    FourierEffect,
    LinearTrendEffect,
    MediaMuEffect,
    MuEffect,
)
from pymc_marketing.mmm.causal_graph import (
    SEASON_NODE,
    TREND_NODE,
    build_direct_effect_fragment,
    build_mmm_star_graph,
    causal_graph_to_graphviz,
    compose_causal_graph,
    host_target_column,
)
from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation
from pymc_marketing.mmm.fourier import YearlyFourier
from pymc_marketing.mmm.linear_trend import LinearTrend
from pymc_marketing.mmm.media_transformation import MediaTransformation
from pymc_marketing.mmm.mmm import MMM, _mu_effect_causal_name


def _minimal_mmm(**kwargs) -> MMM:
    defaults = {
        "date_column": "date",
        "channel_columns": ["channel_1", "channel_2"],
        "target_column": "target",
        "adstock": GeometricAdstock(l_max=4),
        "saturation": LogisticSaturation(),
    }
    defaults.update(kwargs)
    return MMM(**defaults)


def test_star_graph_matches_constructor_args():
    graph = build_mmm_star_graph(
        channel_columns=["tv", "social"],
        control_columns=["price", "holiday"],
        target_column="sales",
        yearly_seasonality=2,
    )
    assert set(graph.nodes) == {
        "tv",
        "social",
        "price",
        "holiday",
        SEASON_NODE,
        "sales",
    }
    assert set(graph.edges) == {
        ("tv", "sales"),
        ("social", "sales"),
        ("price", "sales"),
        ("holiday", "sales"),
        (SEASON_NODE, "sales"),
    }
    assert nx.is_directed_acyclic_graph(graph)


def test_star_graph_without_season_node():
    graph = build_mmm_star_graph(
        channel_columns=["tv"],
        control_columns=None,
        target_column="y",
        yearly_seasonality=None,
    )
    assert SEASON_NODE not in graph.nodes
    assert set(graph.edges) == {("tv", "y")}


def test_mmm_causal_graph_star():
    mmm = _minimal_mmm(
        control_columns=["control_1"],
        yearly_seasonality=2,
    )
    graph = mmm.causal_graph
    assert set(graph.nodes) == {
        "channel_1",
        "channel_2",
        "control_1",
        SEASON_NODE,
        "target",
    }
    assert ("channel_1", "target") in graph.edges
    assert (SEASON_NODE, "target") in graph.edges


def test_mmm_causal_graph_geo_dim_not_a_node():
    mmm = _minimal_mmm(dims=("geo",))
    assert "geo" not in mmm.causal_graph.nodes


def test_causal_graph_rebuild_on_access():
    mmm = _minimal_mmm()
    graph = mmm.causal_graph
    graph.add_edge("channel_1", "channel_2")
    assert ("channel_1", "channel_2") not in mmm.causal_graph.edges


def test_control_mu_effect_fragment_adds_edges():
    mmm = _minimal_mmm(control_columns=["control_1"])
    mmm.add_mu_effect(ControlMuEffect(data_vars=["extra_control"], prefix="extra"))
    graph = mmm.causal_graph
    assert ("extra_control", "target") in graph.edges
    assert ("control_1", "target") in graph.edges


def test_build_direct_effect_fragment():
    graph = build_direct_effect_fragment(["a", "b"], "y")
    assert set(graph.edges) == {("a", "y"), ("b", "y")}


def test_host_target_column():
    mmm = _minimal_mmm()
    assert host_target_column(mmm) == "target"


def test_host_target_column_raises_without_target():
    with pytest.raises(TypeError, match="target_column"):
        host_target_column(object())


def test_fourier_effect_fragment_adds_season_edge():
    mmm = _minimal_mmm(yearly_seasonality=None)
    mmm.add_mu_effect(FourierEffect(fourier=YearlyFourier(n_order=2, prefix="yearly")))
    assert (SEASON_NODE, "target") in mmm.causal_graph.edges


def test_linear_trend_effect_fragment_adds_t_edge():
    mmm = _minimal_mmm(yearly_seasonality=None)
    mmm.add_mu_effect(
        LinearTrendEffect(trend=LinearTrend(n_changepoints=2), prefix="trend")
    )
    assert (TREND_NODE, "target") in mmm.causal_graph.edges


def test_media_mu_effect_fragment_from_xarray_channel_coords():
    mmm = _minimal_mmm(yearly_seasonality=None)
    mmm.xarray_dataset = xr.Dataset(
        {"media_data": (["date", "channel"], np.zeros((5, 2)))},
        coords={"date": np.arange(5), "channel": ["channel_1", "channel_2"]},
    )
    media_effect = MediaMuEffect(
        data_vars=["media_data"],
        media_transformation=MediaTransformation(
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            adstock_first=True,
            dims=("channel",),
        ),
        prefix="media",
    )
    fragment = media_effect.causal_graph_fragment(mmm)
    assert set(fragment.edges) == {
        ("channel_1", "target"),
        ("channel_2", "target"),
    }


def test_media_mu_effect_fragment_falls_back_to_var_name():
    mmm = _minimal_mmm(yearly_seasonality=None)
    media_effect = MediaMuEffect(
        data_vars=["media_product"],
        media_transformation=MediaTransformation(
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            adstock_first=True,
            dims=("product", "channel"),
        ),
        channel_dim="product",
        prefix="media",
    )
    fragment = media_effect.causal_graph_fragment(mmm)
    assert set(fragment.edges) == {("media_product", "target")}


def test_media_mu_effect_fragment_uses_channel_columns_before_build():
    mmm = _minimal_mmm(yearly_seasonality=None)
    media_effect = MediaMuEffect(
        data_vars=["media_data"],
        media_transformation=MediaTransformation(
            adstock=GeometricAdstock(l_max=4),
            saturation=LogisticSaturation(),
            adstock_first=True,
            dims=("channel",),
        ),
        prefix="media",
    )
    fragment = media_effect.causal_graph_fragment(mmm)
    assert set(fragment.edges) == {
        ("channel_1", "target"),
        ("channel_2", "target"),
    }


class _EmptyFragmentMuEffect(MuEffect):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def create_data(self, mmm):
        return None

    def create_effect(self, mmm):
        return None

    def set_data(self, mmm, model, X):
        return None


class _FragmentMuEffect(MuEffect):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    fragment: nx.DiGraph
    prefix: str | None = None

    def create_data(self, mmm):
        return None

    def create_effect(self, mmm):
        return None

    def set_data(self, mmm, model, X):
        return None

    def causal_graph_fragment(self, mmm) -> nx.DiGraph:
        return self.fragment.copy()


def test_compose_union_from_mu_effect_fragment():
    fragment = nx.DiGraph()
    fragment.add_edges_from([("channel_1", "mediator"), ("mediator", "target")])
    mmm = _minimal_mmm()
    mmm.add_mu_effect(_FragmentMuEffect(fragment=fragment))

    graph = mmm.causal_graph
    assert ("channel_1", "mediator") in graph.edges
    assert ("mediator", "target") in graph.edges
    assert ("channel_1", "target") in graph.edges


def test_compose_duplicate_edges_are_no_op():
    fragment = nx.DiGraph()
    fragment.add_edge("channel_1", "target")
    mmm = _minimal_mmm()
    mmm.add_mu_effect(_FragmentMuEffect(fragment=fragment))
    assert ("channel_1", "target") in mmm.causal_graph.edges


def test_mu_effect_default_causal_graph_fragment_is_empty():
    effect = _EmptyFragmentMuEffect()
    assert list(effect.causal_graph_fragment(_minimal_mmm()).nodes) == []


def test_mu_effect_causal_name_includes_prefix():
    effect = _FragmentMuEffect(fragment=nx.DiGraph(), prefix="custom")
    assert _mu_effect_causal_name(effect) == "_FragmentMuEffect('custom')"


def test_compose_causal_graph_raises_on_length_mismatch():
    star = build_mmm_star_graph(["tv"], None, "y", None)
    with pytest.raises(ValueError, match="same length"):
        compose_causal_graph(star, [nx.DiGraph()], effect_names=[])


@patch(
    "pymc_marketing.mmm.causal_graph.nx.is_directed_acyclic_graph",
    return_value=False,
)
def test_build_mmm_star_graph_raises_when_not_dag(_mock_is_dag):
    with pytest.raises(ValueError, match="not a DAG"):
        build_mmm_star_graph(["tv"], None, "y", None)


def test_causal_graph_to_graphviz_raises_without_graphviz(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "graphviz":
            raise ImportError("no graphviz")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match="plot_causal_graph requires"):
        causal_graph_to_graphviz(nx.DiGraph())


def test_compose_cycle_raises_with_effect_name():
    fragment = nx.DiGraph()
    fragment.add_edges_from([("target", "channel_1")])
    mmm = _minimal_mmm()
    mmm.add_mu_effect(_FragmentMuEffect(fragment=fragment))

    with pytest.raises(ValueError, match="_FragmentMuEffect"):
        mmm.causal_graph


def test_compose_cycle_raises_with_prefixed_effect_name():
    fragment = nx.DiGraph()
    fragment.add_edge("target", "channel_1")
    mmm = _minimal_mmm()
    mmm.add_mu_effect(_FragmentMuEffect(fragment=fragment, prefix="custom"))

    with pytest.raises(ValueError, match="_FragmentMuEffect\\('custom'\\)"):
        _ = mmm.causal_graph


def test_no_causal_graphical_model_without_dag_kwarg():
    mmm = _minimal_mmm()
    assert not hasattr(mmm, "causal_graphical_model")


def test_build_star_graph_raises_when_target_matches_predictor():
    with pytest.raises(ValueError, match="target_column"):
        build_mmm_star_graph(
            channel_columns=["y"],
            control_columns=None,
            target_column="y",
            yearly_seasonality=None,
        )


def test_build_star_graph_raises_when_season_column_conflicts_with_fourier():
    with pytest.raises(ValueError, match="yearly_seasonality"):
        build_mmm_star_graph(
            channel_columns=["tv"],
            control_columns=["season"],
            target_column="y",
            yearly_seasonality=2,
        )


def test_plot_causal_graph_returns_graphviz_digraph():
    mmm = _minimal_mmm(control_columns=["control_1"], yearly_seasonality=2)
    digraph = mmm.plot_causal_graph()
    assert isinstance(digraph, graphviz.Digraph)
    body = "".join(digraph.body)
    for node in ("channel_1", "channel_2", "control_1", SEASON_NODE, "target"):
        assert node in body


def test_plot_causal_graph_passes_rankdir():
    mmm = _minimal_mmm()
    digraph = mmm.plot_causal_graph(rankdir="TB")
    assert "rankdir=TB" in digraph.source
