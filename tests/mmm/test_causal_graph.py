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

import graphviz
import networkx as nx
import pytest
from pydantic import ConfigDict

from pymc_marketing.mmm.additive_effect import ControlMuEffect, MuEffect
from pymc_marketing.mmm.causal_graph import (
    SEASON_NODE,
    build_mmm_star_graph,
)
from pymc_marketing.mmm.components.adstock import GeometricAdstock
from pymc_marketing.mmm.components.saturation import LogisticSaturation
from pymc_marketing.mmm.mmm import MMM


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


def test_add_mu_effect_with_empty_fragment_does_not_change_graph():
    mmm = _minimal_mmm(control_columns=["control_1"])
    before = set(mmm.causal_graph.edges)
    mmm.add_mu_effect(ControlMuEffect(data_vars=["extra_control"], prefix="extra"))
    assert set(mmm.causal_graph.edges) == before


class _FragmentMuEffect(MuEffect):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    fragment: nx.DiGraph

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


def test_compose_cycle_raises_with_effect_name():
    fragment = nx.DiGraph()
    fragment.add_edges_from([("target", "channel_1")])
    mmm = _minimal_mmm()
    mmm.add_mu_effect(_FragmentMuEffect(fragment=fragment))

    with pytest.raises(ValueError, match="_FragmentMuEffect"):
        mmm.causal_graph


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


def test_plot_causal_graph_returns_graphviz_digraph():
    mmm = _minimal_mmm(control_columns=["control_1"], yearly_seasonality=2)
    digraph = mmm.plot_causal_graph()
    assert isinstance(digraph, graphviz.Digraph)
    body = "".join(digraph.body)
    for node in ("channel_1", "channel_2", "control_1", SEASON_NODE, "target"):
        assert node in body
