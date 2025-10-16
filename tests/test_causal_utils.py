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
import pydot  # type: ignore[import]
import pytest  # type: ignore[import]

from pymc_marketing.causal_utils import (
    _label_of,
    _parse_dot_CPdag,
    _skeleton,
    _v_structures,
    same_markov_equivalence_class_CPdag,
)


@pytest.mark.parametrize(
    "node, expected",
    [
        (pydot.Node("n1", label="lbl"), "lbl"),
        (pydot.Node("n2"), "n2"),
    ],
)
def test_label_of(node, expected):
    assert _label_of(node) == expected


@pytest.mark.parametrize(
    "dot_input, exp_nodes, exp_de, exp_ue",
    [
        (
            "digraph { A; B; A -> B }",
            {"A", "B"},
            {("A", "B")},
            set(),
        ),
        (
            "digraph { A -> C; B -> C }",
            {"A", "B", "C"},
            {("A", "C"), ("B", "C")},
            set(),
        ),
    ],
)
def test_parse_dot_CPdag(dot_input, exp_nodes, exp_de, exp_ue):
    nodes, de, ue = _parse_dot_CPdag(dot_input)
    assert nodes == exp_nodes
    assert de == exp_de
    assert ue == exp_ue


@pytest.mark.parametrize(
    "dot_input, exp_nodes, exp_de, exp_ue",
    [
        (
            "strict graph { A -- B }",
            {"A", "B"},
            set(),
            {frozenset(("A", "B"))},
        ),
    ],
)
def test_parse_dot_CPdag_strict_graph(dot_input, exp_nodes, exp_de, exp_ue):
    nodes, de, ue = _parse_dot_CPdag(dot_input)
    assert nodes == exp_nodes
    assert de == exp_de
    assert ue == exp_ue


@pytest.mark.parametrize(
    "dot_input, exp_nodes, exp_de, exp_ue",
    [
        (
            "graph { A -- B }",
            {"A", "B"},
            set(),
            {frozenset(("A", "B"))},
        ),
    ],
)
def test_parse_dot_CPdag_undirected_edges(dot_input, exp_nodes, exp_de, exp_ue):
    nodes, de, ue = _parse_dot_CPdag(dot_input)
    assert nodes == exp_nodes
    assert de == exp_de
    assert ue == exp_ue


def test_parse_dot_CPdag_invalid_type_raises_TypeError():
    with pytest.raises(TypeError):
        _parse_dot_CPdag(123)  # type: ignore[arg-type]


def test_parse_dot_CPdag_unparseable_raises_ValueError():
    bad_dot = ""
    # empty string yields no graphs
    with pytest.raises(ValueError):
        _parse_dot_CPdag(bad_dot)


@pytest.mark.parametrize(
    "dir_edges, undirected_edges, expected_skeleton",
    [
        (
            {("A", "B")},
            {frozenset(("B", "C"))},
            {frozenset(("A", "B")), frozenset(("B", "C"))},
        ),
        (
            {("X", "Y"), ("Y", "Z")},
            set(),
            {frozenset(("X", "Y")), frozenset(("Y", "Z"))},
        ),
    ],
)
def test_skeleton(dir_edges, undirected_edges, expected_skeleton):
    # nodes argument is not used internally
    result = _skeleton(set(), dir_edges, undirected_edges)
    assert result == expected_skeleton


def test_skeleton_empty():
    assert _skeleton(set(), set(), set()) == set()


@pytest.mark.parametrize(
    "dir_edges, skeleton, expected_v",
    [
        (
            {("A", "C"), ("B", "C")},
            {frozenset(("A", "C")), frozenset(("B", "C"))},
            {(("A", "B"), "C")},
        ),
        (
            {("D", "E"), ("E", "F"), ("D", "F")},
            {frozenset(("D", "E")), frozenset(("E", "F")), frozenset(("D", "F"))},
            set(),  # no unshielded collider
        ),
    ],
)
def test_v_structures(dir_edges, skeleton, expected_v):
    result = _v_structures(dir_edges, skeleton)
    assert result == expected_v


def test_v_structures_empty():
    assert _v_structures(set(), set()) == set()


def test_parse_dot_CPdag_skip_style_node_named_edge():
    # Node named 'edge' should be skipped in id_to_label
    dot = "digraph { edge; A; edge -> A }"
    nodes, de, ue = _parse_dot_CPdag(dot)
    # Only A should map via id_to_label; 'edge' should be present from edges but label mapping skipped
    assert "A" in nodes
    # directed edge from 'edge' to 'A'
    assert ("edge", "A") in de
    assert frozenset(("edge", "A")) not in ue


def test_same_markov_equivalence_class_CPdag_true():
    # Identical DAGs should be equivalent
    g1 = pydot.Dot(graph_type="digraph")
    g1.add_node(pydot.Node("A"))
    g1.add_node(pydot.Node("B"))
    g1.add_edge(pydot.Edge("A", "B"))

    g2 = pydot.Dot(graph_type="digraph")
    # reversed addition order
    g2.add_edge(pydot.Edge("A", "B"))
    g2.add_node(pydot.Node("B"))
    g2.add_node(pydot.Node("A"))

    # pass DOT text to function
    assert same_markov_equivalence_class_CPdag(g1.to_string(), g2.to_string())


def test_same_markov_equivalence_class_CPdag_false_vstructure_shielded():
    # Graph with v-structure vs shielded one
    g1 = pydot.Dot(graph_type="digraph")
    g1.add_edge(pydot.Edge("A", "C"))
    g1.add_edge(pydot.Edge("B", "C"))

    g2 = pydot.Dot(graph_type="digraph")
    g2.add_edge(pydot.Edge("A", "C"))
    g2.add_edge(pydot.Edge("B", "C"))
    # add undirected edge between A and B to shield v-structure
    g2.add_edge(pydot.Edge("A", "B", dir="none"))

    # pass DOT text to function
    assert not same_markov_equivalence_class_CPdag(g1.to_string(), g2.to_string())


def test_same_markov_equivalence_class_CPdag_with_strings():
    # identical skeleton and no directed edges
    g1 = "graph { A -- B }"
    g2 = "graph { B -- A }"
    assert same_markov_equivalence_class_CPdag(g1, g2)


def test_same_markov_equivalence_class_CPdag_false_diff_nodes():
    g1 = "digraph { A }"
    g2 = "digraph { B }"
    assert not same_markov_equivalence_class_CPdag(g1, g2)


def test_same_markov_equivalence_class_CPdag_invalid_type():
    with pytest.raises(TypeError):
        same_markov_equivalence_class_CPdag(123, "graph { A -- B }")  # type: ignore[arg-type]


def test_same_markov_equivalence_class_CPdag_unparseable_dot():
    # empty strings lead to ValueError in parsing
    with pytest.raises(ValueError):
        same_markov_equivalence_class_CPdag("", "")


# Corner cases for _parse_dot_CPdag
class FakeSource:
    def __init__(self, source: str):
        self.source = source


def test_parse_dot_CPdag_from_source_protocol():
    src = "graph { X -- Y }"
    fs = FakeSource(src)
    nodes, de, ue = _parse_dot_CPdag(fs)
    assert nodes == {"X", "Y"}
    assert de == set()
    assert ue == {frozenset(("X", "Y"))}


def test_parse_dot_CPdag_dir_none_attribute():
    dot = "digraph { A -> B [dir=none] }"
    nodes, de, ue = _parse_dot_CPdag(dot)
    assert nodes == {"A", "B"}
    assert de == set()
    assert ue == {frozenset(("A", "B"))}


def test_parse_dot_CPdag_skip_node_and_graph_style():
    # Nodes named 'node' and 'graph' should be skipped
    dot = "digraph { graph; node; A -> B }"
    nodes, de, ue = _parse_dot_CPdag(dot)
    assert "graph" not in nodes
    assert "node" not in nodes
    assert {"A", "B"} <= nodes
    assert de == {("A", "B")}
    assert ue == set()
