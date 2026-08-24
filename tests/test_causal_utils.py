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
"""Tests for :mod:`pymc_marketing.causal_utils`.

``same_markov_equivalence_class_CPdag`` delegates to pathmc's
``same_markov_equivalence_class``; the full Markov-equivalence behavior (DOT
parsing, skeleton + unshielded-collider equality, corner cases) is exhaustively
tested in pathmc. What is checked here is that the thin wrapper delegates
correctly and stays importable without the optional ``pathmc`` dependency.
"""

import sys

import pytest

from pymc_marketing.causal_utils import same_markov_equivalence_class_CPdag


def test_chain_and_fork_are_equivalent():
    # Chain A->B->C and fork A<-B->C share a skeleton with no v-structure.
    assert (
        same_markov_equivalence_class_CPdag(
            "digraph { A -> B; B -> C; }",
            "digraph { B -> A; B -> C; }",
        )
        is True
    )


def test_collider_is_not_equivalent_to_chain():
    assert (
        same_markov_equivalence_class_CPdag(
            "digraph { A -> B; B -> C; }",
            "digraph { A -> B; C -> B; }",
        )
        is False
    )


def test_undirected_graph_strings_order_independent():
    assert (
        same_markov_equivalence_class_CPdag("graph { A -- B }", "graph { B -- A }")
        is True
    )


def test_different_node_sets_are_not_equivalent():
    assert (
        same_markov_equivalence_class_CPdag("digraph { A }", "digraph { B }") is False
    )


def test_accepts_objects_exposing_source():
    class Digraph:
        def __init__(self, source: str) -> None:
            self.source = source

    assert (
        same_markov_equivalence_class_CPdag(
            Digraph("digraph { A -> B; B -> C; }"),
            Digraph("digraph { B -> A; B -> C; }"),
        )
        is True
    )


def test_accepts_graphviz_digraph_with_comment():
    """``Digraph(comment=...)`` puts ``// <comment>`` above the DOT header."""
    from graphviz import Digraph

    chain = Digraph(comment="True Causal DAG")
    for tail, head in [("A", "B"), ("B", "C")]:
        chain.edge(tail, head)
    fork = Digraph()
    for tail, head in [("B", "A"), ("B", "C")]:
        fork.edge(tail, head)

    assert chain.source.startswith("// True Causal DAG")
    assert same_markov_equivalence_class_CPdag(chain.source, fork.source) is True
    assert same_markov_equivalence_class_CPdag(chain, fork) is True


@pytest.mark.parametrize(
    "preamble",
    ["", "// leading digraph comment\n", "/* block\n   comment */\n", "# hash\n"],
    ids=["none", "line", "block", "hash"],
)
def test_dot_preamble_is_ignored(preamble):
    assert (
        same_markov_equivalence_class_CPdag(
            f"{preamble}digraph {{ A -> B; B -> C; }}",
            "digraph { B -> A; B -> C; }",
        )
        is True
    )


def test_accepts_networkx_digraphs():
    """pathmc compares ``networkx`` graphs directly; they pass through untouched."""
    import networkx as nx

    chain = nx.DiGraph([("A", "B"), ("B", "C")])
    fork = nx.DiGraph([("B", "A"), ("B", "C")])
    collider = nx.DiGraph([("A", "B"), ("C", "B")])

    assert same_markov_equivalence_class_CPdag(chain, fork) is True
    assert same_markov_equivalence_class_CPdag(chain, collider) is False


def test_errors_actionably_without_pathmc(monkeypatch):
    """``None`` in ``sys.modules`` is what a missing ``dag`` extra looks like."""
    monkeypatch.setitem(sys.modules, "pathmc", None)
    with pytest.raises(ImportError, match=r"pip install pymc-marketing\[dag\]"):
        same_markov_equivalence_class_CPdag("digraph { A }", "digraph { A }")
