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
"""A moducle with utilities for causal reasoning and discovery."""

from itertools import combinations
from typing import Protocol

import pydot  # type: ignore[import]


class _SupportsSource(Protocol):
    @property
    def source(self) -> str: ...


def _label_of(pnode: pydot.Node) -> str:
    # prefer 'label' if present; else use the DOT node name
    lbl = pnode.get_attributes().get("label")
    return lbl if lbl is not None else pnode.get_name().strip('"')


def _parse_dot_CPdag(
    dot_input: str | _SupportsSource,
) -> tuple[set[str], set[tuple[str, str]], set[frozenset[str]]]:
    """Parse DOT (string or graphviz.Digraph) into components."""
    if hasattr(dot_input, "source"):
        dot_text = dot_input.source
    elif isinstance(dot_input, str):
        dot_text = dot_input
    else:
        raise TypeError("dot_input must be DOT text or graphviz.Digraph")

    graphs = pydot.graph_from_dot_data(dot_text)
    if not graphs:
        raise ValueError("Could not parse DOT text.")
    pdg = graphs[0]

    # map DOT node IDs -> normalized labels
    id_to_label: dict[str, str] = {}
    for pnode in pdg.get_nodes():
        name = pnode.get_name().strip('"')
        if name in ("graph", "node", "edge"):  # pydot includes style nodes sometimes
            continue
        id_to_label[name] = _label_of(pnode)

    nodes: set[str] = set(id_to_label.values())

    dir_edges: set[tuple[str, str]] = set()  # for v-structures
    undirected_edges: set[frozenset[str]] = set()  # CPDAG-style undirected edges

    # detect if this is an undirected graph (Graph or strict Graph) vs directed (Digraph)
    type_lower = pdg.get_type().lower().replace("strict ", "")
    is_undirected_graph = type_lower == "graph"
    for e in pdg.get_edges():
        u_id = e.get_source().strip('"')
        v_id = e.get_destination().strip('"')
        # If an endpoint didn't appear as a standalone node, infer its label from ID
        u = id_to_label.get(u_id, u_id)
        v = id_to_label.get(v_id, v_id)
        if is_undirected_graph:
            # all edges in a Graph are undirected
            if u != v:
                undirected_edges.add(frozenset((u, v)))
        else:
            attrs = {k.lower(): v for k, v in e.get_attributes().items()}
            edge_dir = attrs.get("dir", None)  # 'none' means undirected in CPDAG sense
            if edge_dir == "none":
                if u != v:
                    undirected_edges.add(frozenset((u, v)))
            else:
                if u != v:
                    dir_edges.add((u, v))

        # ensure nodes exist even if only on edges
        nodes.add(u)
        nodes.add(v)

    return nodes, dir_edges, undirected_edges


def _skeleton(
    nodes: set[str],
    dir_edges: set[tuple[str, str]],
    undirected_edges: set[frozenset[str]],
) -> set[frozenset[str]]:
    """Return the undirected skeleton: all adjacencies regardless of orientation."""
    S = set(undirected_edges)
    for u, v in dir_edges:
        S.add(frozenset((u, v)))
    return S


def _v_structures(
    dir_edges: set[tuple[str, str]],
    skeleton_undirected: set[frozenset[str]],
) -> set[tuple[tuple[str, str], str]]:
    """Identify unshielded colliders in the CPDAG."""
    # parents map from directed edges
    parents: dict[str, set[str]] = {}
    for a, c in dir_edges:
        parents.setdefault(c, set()).add(a)

    vstructs: set[tuple[tuple[str, str], str]] = set()
    for c, pars in parents.items():
        if len(pars) < 2:
            continue
        for a, b in combinations(pars, 2):
            if frozenset((a, b)) not in skeleton_undirected:  # unshielded
                ordered = (a, b) if a < b else (b, a)
                vstructs.add((ordered, c))
    return vstructs


def same_markov_equivalence_class_CPdag(
    dot1: str | _SupportsSource, dot2: str | _SupportsSource
) -> bool:
    """Determine whether two DOT graphs share a Markov equivalence class."""
    n1, de1, ue1 = _parse_dot_CPdag(dot1)
    n2, de2, ue2 = _parse_dot_CPdag(dot2)

    # Compare node sets by labels
    if n1 != n2:
        return False

    S1 = _skeleton(n1, de1, ue1)
    S2 = _skeleton(n2, de2, ue2)
    if S1 != S2:
        return False

    V1 = _v_structures(de1, S1)
    V2 = _v_structures(de2, S2)
    return V1 == V2
