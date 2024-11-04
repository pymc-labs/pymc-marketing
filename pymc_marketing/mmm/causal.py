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
"""Causal identification class."""

from collections.abc import Sequence
from typing import Any

import networkx as nx
from dowhy.causal_identifier.auto_identifier import (
    construct_backdoor_estimand,
    construct_frontdoor_estimand,
)
from dowhy.graph import (
    build_graph,
    build_graph_from_str,
    get_backdoor_paths,
    get_instruments,
)


class CausalGraphModel:
    """Represent a causal model based on a Directed Acyclic Graph (DAG).

    Provides methods to analyze causal relationships, determine adjustment sets,
    and assess the possibility of backdoor and frontdoor adjustments.

    Parameters
    ----------
    graph : nx.DiGraph
        A directed acyclic graph representing the causal relationships among variables.
    treatment : list[str]
        A list of treatment variable names.
    outcome : list[str]
        A list of outcome variable names.
    """

    def __init__(self, graph: nx.DiGraph, treatment: list[str], outcome: list[str]):
        self.graph = graph
        self.treatment = treatment
        self.outcome = outcome

    @classmethod
    def from_string(cls, graph_str: str, treatment: list[str], outcome: list[str]):
        """Create a CausalGraphModel from a string representation of a graph.

        Parameters
        ----------
        graph_str : str
            A string representation of the graph (e.g., in DOT format).
        treatment : list[str]
            A list of treatment variable names.
        outcome : list[str]
            A list of outcome variable names.

        Returns
        -------
        CausalGraphModel
            An instance of CausalGraphModel constructed from the given graph string.
        """
        graph = build_graph_from_str(graph_str)
        return cls(graph, treatment, outcome)

    @classmethod
    def from_nodes_and_edges(
        cls,
        action_nodes: list[str],
        outcome_nodes: list[str],
        common_cause_nodes: list[str] | None = None,
        instrument_nodes: list[str] | None = None,
        mediator_nodes: list[str] | None = None,
    ):
        """Create a CausalGraphModel from lists of nodes categorized by their roles in the causal graph.

        Parameters
        ----------
        action_nodes : list[str]
            List of treatment (action) variable names.
        outcome_nodes : list[str]
            List of outcome variable names.
        common_cause_nodes : Optional[list[str]], default=None
            List of common cause (confounder) variable names.
        instrument_nodes : Optional[list[str]], default=None
            List of instrumental variable names.
        mediator_nodes : Optional[list[str]], default=None
            List of mediator variable names.

        Returns
        -------
        CausalGraphModel
            An instance of CausalGraphModel constructed from the specified nodes.
        """
        graph = build_graph(
            action_nodes=action_nodes,
            outcome_nodes=outcome_nodes,
            common_cause_nodes=common_cause_nodes,
            instrument_nodes=instrument_nodes,
            mediator_nodes=mediator_nodes,
        )
        return cls(graph, action_nodes, outcome_nodes)

    def get_backdoor_paths(self) -> dict[str, dict[str, Sequence[Any]]]:
        """Find all backdoor paths between treatment and outcome variables and compute adjustment sets.

        Returns
        -------
        dict[str, dict[str, Sequence[Any]]]
            A dictionary where each key is a treatment variable, and the value is another dictionary containing:
            - 'adjustment_sets': A list of adjustment sets (lists of variable names) for backdoor adjustment.
            - 'minimal_adjustment_set': The minimal adjustment set (with the least number of variables) required
              to block all backdoor paths.
        """
        backdoor_dict = {}
        for treatment_node in self.treatment:
            paths = get_backdoor_paths(self.graph, [treatment_node], self.outcome)

            # Exclude treatment and outcome nodes from each backdoor path to obtain valid adjustment sets
            adjustment_sets = {
                tuple(sorted(set(path) - {treatment_node} - set(self.outcome)))
                for path in paths
            }

            backdoor_dict[treatment_node] = {
                "adjustment_sets": [
                    list(adjustment_set) for adjustment_set in adjustment_sets
                ],
                "minimal_adjustment_set": min(adjustment_sets, key=len)
                if adjustment_sets
                else [],
            }

        return backdoor_dict

    def is_backdoor_adjustment_possible(self) -> bool:
        """Determine whether backdoor adjustment is possible for the causal model.

        Returns
        -------
        bool
            True if backdoor adjustment is possible (i.e., there exists a backdoor path), False otherwise.
        """
        backdoor_paths = self.get_backdoor_paths()
        return any(
            backdoor_paths[node]["minimal_adjustment_set"] for node in backdoor_paths
        )

    def get_minimal_adjustment_sets(self) -> set[str] | None:
        """Compute the minimal adjustment set(s) required for backdoor adjustment using DoWhy.

        Returns
        -------
        Optional[Set[str]]
            A set of variable names representing the minimal adjustment set, or None if not identifiable.
        """
        try:
            estimand = construct_backdoor_estimand(
                self.graph, self.treatment[0], self.outcome[0]
            )
            return estimand.get_backdoor_variables()
        except Exception as e:
            print("Error identifying backdoor adjustment set:", e)
            return None

    def is_frontdoor_adjustment_possible(self) -> bool:
        """Determine whether frontdoor adjustment is possible for the causal model.

        Returns
        -------
        bool
            True if frontdoor adjustment is possible, False otherwise.
        """
        try:
            frontdoor_estimand = construct_frontdoor_estimand(
                self.graph, self.treatment[0], self.outcome[0]
            )
            return frontdoor_estimand is not None
        except Exception:
            return False

    def get_instrumental_variables(self):
        """Identify instrumental variables in the causal graph using DoWhy.

        Returns
        -------
        list[str]
            A list of variable names that are instrumental variables, or an empty list if none are found.
        """
        try:
            instruments = get_instruments(self.graph, self.treatment, self.outcome)
            return instruments
        except Exception as e:
            print("Error identifying instruments:", e)
            return []

    def get_unique_minimal_adjustment_elements(self):
        """Extract unique variables from all minimal adjustment sets across all treatments.

        Returns
        -------
        Set[str]
            A set of unique variable names that are part of the minimal adjustment sets.
        """
        backdoor_info = self.get_backdoor_paths()
        unique_elements = set()
        for _node, info in backdoor_info.items():
            unique_elements.update(info["minimal_adjustment_set"])
        return unique_elements
