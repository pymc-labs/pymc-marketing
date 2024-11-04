import networkx as nx
from dowhy.graph import get_backdoor_paths, get_instruments, build_graph_from_str, build_graph
from dowhy.causal_identifier.auto_identifier import (
    construct_backdoor_estimand,
    construct_frontdoor_estimand
)
from typing import list, Set, Optional

class CausalGraphModel:
    """
    A class representing a causal model based on a Directed Acyclic Graph (DAG).
    It provides methods to analyze causal relationships, determine adjustment sets,
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
        """
        Constructs a CausalModel from a string representation of a graph.

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
        CausalModel
            An instance of CausalModel constructed from the given graph string.
        """
        graph = build_graph_from_str(graph_str)
        return cls(graph, treatment, outcome)

    @classmethod
    def from_nodes_and_edges(cls, action_nodes: list[str], outcome_nodes: list[str],
                             common_cause_nodes: Optional[list[str]] = None,
                             instrument_nodes: Optional[list[str]] = None,
                             mediator_nodes: Optional[list[str]] = None):
        """
        Constructs a CausalModel from lists of nodes categorized by their roles in the causal graph.

        Parameters
        ----------
        action_nodes : list[str]
            list of treatment (action) variable names.
        outcome_nodes : list[str]
            list of outcome variable names.
        common_cause_nodes : Optional[list[str]], default=None
            list of common cause (confounder) variable names.
        instrument_nodes : Optional[list[str]], default=None
            list of instrumental variable names.
        mediator_nodes : Optional[list[str]], default=None
            list of mediator variable names.

        Returns
        -------
        CausalModel
            An instance of CausalModel constructed from the specified nodes.
        """
        graph = build_graph(
            action_nodes=action_nodes,
            outcome_nodes=outcome_nodes,
            common_cause_nodes=common_cause_nodes,
            instrument_nodes=instrument_nodes,
            mediator_nodes=mediator_nodes
        )
        return cls(graph, action_nodes, outcome_nodes)

    def get_backdoor_paths(self) -> dict[str, dict[str, list[list[str]]]]:
      """
        Finds all backdoor paths between treatment and outcome variables and computes adjustment sets.

        Returns
        -------
        dict[str, dict[str, list[list[str]]]]
            A dictionary where each key is a treatment variable, and the value is another dictionary containing:
            - 'adjustment_sets': A list of adjustment sets (lists of variable names) for backdoor adjustment.
            - 'minimal_adjustment_set': The minimal adjustment set (with the least number of variables) required to block all backdoor paths.
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
              "adjustment_sets": [list(adjustment_set) for adjustment_set in adjustment_sets],
              "minimal_adjustment_set": min(adjustment_sets, key=len) if adjustment_sets else []
          }

      return backdoor_dict

    def is_backdoor_adjustment_possible(self) -> bool:
        """
        Determines whether backdoor adjustment is possible for the causal model.

        Returns
        -------
        bool
            True if backdoor adjustment is possible (i.e., there exists a backdoor path), False otherwise.
        """
        backdoor_paths = self.get_backdoor_paths()
        return any(backdoor_paths[node]["minimal_adjustment_set"] for node in backdoor_paths)

    def get_minimal_adjustment_sets(self) -> Optional[Set[str]]:
        """
        Computes the minimal adjustment set(s) required for backdoor adjustment using DoWhy.

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
        """
        Determines whether frontdoor adjustment is possible for the causal model.

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

    def get_instrumental_variables(self) -> list[str]:
        """
        Identifies instrumental variables in the causal graph using DoWhy.

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

    def get_unique_minimal_adjustment_elements(self) -> Set[str]:
        """
        Extracts unique variables from all minimal adjustment sets across all treatments.

        Returns
        -------
        Set[str]
            A set of unique variable names that are part of the minimal adjustment sets.
        """
        backdoor_info = self.get_backdoor_paths()
        unique_elements = set()
        for node, info in backdoor_info.items():
            unique_elements.update(info["minimal_adjustment_set"])
        return unique_elements
