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
"""Model-implied causal DAG construction for Media Mix Models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    import graphviz

SEASON_NODE = "season"


def build_mmm_star_graph(
    channel_columns: list[str],
    control_columns: list[str] | None,
    target_column: str,
    yearly_seasonality: int | None,
) -> nx.DiGraph:
    """Build the default MMM causal star: channels, controls, and season → target.

    Parameters
    ----------
    channel_columns
        Media channel column names.
    control_columns
        Control column names, if any.
    target_column
        Outcome column name.
    yearly_seasonality
        When set, a ``season`` node is included with an edge to the target.

    Returns
    -------
    networkx.DiGraph
        Directed acyclic graph for the implicit star assumption.

    Raises
    ------
    ValueError
        If a predictor column name equals the target column name, or if
        ``yearly_seasonality`` is set while a channel or control is named
        ``season``.
    """
    predictors = list(channel_columns)
    if control_columns:
        predictors.extend(control_columns)
    if yearly_seasonality is not None:
        if SEASON_NODE in predictors:
            raise ValueError(
                f"yearly_seasonality cannot be used when a channel or control "
                f"column is named {SEASON_NODE!r}."
            )
        predictors.append(SEASON_NODE)

    if target_column in predictors:
        raise ValueError(
            f"target_column {target_column!r} must not match a channel, "
            "control, or season node name."
        )

    graph = nx.DiGraph()
    graph.add_node(target_column)
    for predictor in predictors:
        graph.add_node(predictor)
        graph.add_edge(predictor, target_column)

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("MMM star graph is not a DAG.")

    return graph


def compose_causal_graph(
    star: nx.DiGraph,
    fragments: list[nx.DiGraph],
    *,
    effect_names: list[str],
) -> nx.DiGraph:
    """Union the MMM star with causal fragments from attached mu effects.

    Parameters
    ----------
    star
        Base star graph from :func:`build_mmm_star_graph`.
    fragments
        Per-effect causal fragments.
    effect_names
        Names used when reporting which effect introduced a cycle.

    Returns
    -------
    networkx.DiGraph
        Union of ``star`` and all fragments.

    Raises
    ------
    ValueError
        If ``fragments`` and ``effect_names`` differ in length, or a fragment
        introduces a cycle.
    """
    if len(fragments) != len(effect_names):
        raise ValueError(
            "fragments and effect_names must have the same length: "
            f"got {len(fragments)} fragments and {len(effect_names)} names."
        )

    composed = star.copy()
    for fragment, effect_name in zip(fragments, effect_names, strict=True):
        composed = nx.compose(composed, fragment)
        if not nx.is_directed_acyclic_graph(composed):
            raise ValueError(
                f"MuEffect {effect_name!r} introduces a cycle in the causal DAG."
            )

    return composed


def causal_graph_to_graphviz(
    causal_graph: nx.DiGraph,
    *,
    rankdir: str = "LR",
) -> graphviz.Digraph:
    """Convert a causal DAG to a graphviz Digraph for plotting.

    Parameters
    ----------
    causal_graph
        Model-implied causal graph.
    rankdir
        Graph layout direction passed to graphviz.

    Returns
    -------
    graphviz.Digraph

    Raises
    ------
    ImportError
        If the optional ``graphviz`` package is not installed.
    """
    try:
        import graphviz
    except ImportError as exc:
        raise ImportError(
            "plot_causal_graph requires the graphviz Python package and the "
            "Graphviz system binaries. Install with: pip install graphviz"
        ) from exc

    digraph = graphviz.Digraph()
    digraph.attr(rankdir=rankdir)
    for node in causal_graph.nodes:
        digraph.node(node)
    for parent, child in causal_graph.edges:
        digraph.edge(parent, child)
    return digraph
