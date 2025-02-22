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
"""Causal identification class."""

import copy
import itertools
import warnings
from abc import abstractmethod

import arviz as az
import networkx as nx
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from graphviz import Digraph
from pydantic import BaseModel, Field, model_validator

try:
    from dowhy import CausalModel
except ImportError:

    class LazyCausalModel:
        """Lazy import of dowhy's CausalModel."""

        def __init__(self, *args, **kwargs):
            msg = (
                "To use Causal Graph functionality, please install the optional dependencies with: "
                "pip install pymc-marketing[dag]"
            )
            raise ImportError(msg)

    CausalModel = LazyCausalModel


class CausalGraphModel:
    """Represent a causal model based on a Directed Acyclic Graph (DAG).

    Provides methods to analyze causal relationships and determine the minimal adjustment set
    for backdoor adjustment between treatment and outcome variables.

    Parameters
    ----------
    causal_model : CausalModel
        An instance of dowhy's CausalModel, representing the causal graph and its relationships.
    treatment : list[str]
        A list of treatment variable names.
    outcome : str
        The outcome variable name.

    References
    ----------
    .. [1] https://github.com/microsoft/dowhy
    """

    def __init__(
        self, causal_model: CausalModel, treatment: list[str] | tuple[str], outcome: str
    ) -> None:
        self.causal_model = causal_model
        self.treatment = treatment
        self.outcome = outcome

    @classmethod
    def build_graphical_model(
        cls, graph: str, treatment: list[str] | tuple[str], outcome: str
    ) -> "CausalGraphModel":
        """Create a CausalGraphModel from a string representation of a graph.

        Parameters
        ----------
        graph : str
            A string representation of the graph (e.g., String in DOT format).
        treatment : list[str]
            A list of treatment variable names.
        outcome : str
            The outcome variable name.

        Returns
        -------
        CausalGraphModel
            An instance of CausalGraphModel constructed from the given graph string.
        """
        causal_model = CausalModel(
            data=pd.DataFrame(), graph=graph, treatment=treatment, outcome=outcome
        )
        return cls(causal_model, treatment, outcome)

    def get_backdoor_paths(self) -> list[list[str]]:
        """Find all backdoor paths between the combined treatment and outcome variables.

        Returns
        -------
        list[list[str]]
            A list of backdoor paths, where each path is represented as a list of variable names.

        References
        ----------
        .. [1] Causal Inference in Statistics: A Primer
        By Judea Pearl, Madelyn Glymour, Nicholas P. Jewell · 2016
        """
        # Use DoWhy's internal method to get backdoor paths for all treatments combined
        return self.causal_model._graph.get_backdoor_paths(
            nodes1=self.treatment, nodes2=[self.outcome]
        )

    def get_unique_adjustment_nodes(self) -> list[str]:
        """Compute the minimal adjustment set required for backdoor adjustment across all treatments.

        Returns
        -------
        list[str]
            A list of unique adjustment variables needed to block all backdoor paths.
        """
        paths = self.get_backdoor_paths()
        # Flatten paths and exclude treatments and outcome from adjustment set
        adjustment_nodes = set(
            node
            for path in paths
            for node in path
            if node not in self.treatment and node != self.outcome
        )
        return list(adjustment_nodes)

    def compute_adjustment_sets(
        self,
        channel_columns: list[str] | tuple[str],
        control_columns: list[str] | None = None,
    ) -> list[str] | None:
        """Compute minimal adjustment sets and handle warnings."""
        channel_columns = list(channel_columns)
        if control_columns is None:
            return control_columns

        self.adjustment_set = self.get_unique_adjustment_nodes()

        common_controls = set(control_columns).intersection(self.adjustment_set)
        unique_controls = set(control_columns) - set(self.adjustment_set)

        if unique_controls:
            warnings.warn(
                f"Columns {unique_controls} are not in the adjustment set. Controls are being modified.",
                stacklevel=2,
            )

        control_columns = list(common_controls - set(channel_columns))

        self.minimal_adjustment_set = control_columns + list(channel_columns)

        for column in self.adjustment_set:
            if column not in control_columns and column not in channel_columns:
                warnings.warn(
                    f"""Column {column} in adjustment set not found in data.
                    Not controlling for this may induce bias in treatment effect estimates.""",
                    stacklevel=2,
                )

        return control_columns


class BayesianCausalDiscoveryBase(BaseModel):
    """
    Base class for Bayesian causal discovery algorithms.

    This class provides common functionality for model building, sampling, and graph utilities.
    Parameters are validated with Pydantic.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing variables as columns
    penalty_weight : float, optional
        Weight factor for penalizing model complexity, by default 1.0
    no_descendants : List[str], optional
        List of variables known to have no descendants, by default None
    no_ascendents : List[str], optional
        List of variables known to have no ancestors, by default None

    Attributes
    ----------
    nodes : List[str]
        List of variable names from data
    graph : Dict[str, List[str]]
        Current graph structure as adjacency list
    local_score_cache : Dict
        Cache for computed local scores
    inference_cache : Dict
        Cache for inference results
    sample_kwargs : Dict
        Additional sampling parameters
    """

    data: pd.DataFrame
    penalty_weight: float = 1.0
    no_descendants: list[str] | None = None
    no_ascendents: list[str] | None = None

    # Internal fields; these are set in __init__
    nodes: list[str] = Field(default_factory=list)
    graph: dict[str, list[str]] = Field(default_factory=dict)
    local_score_cache: dict = Field(default_factory=dict)
    inference_cache: dict = Field(default_factory=dict)
    sample_kwargs: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    def set_defaults(cls, values):
        """
        Set default values for optional parameters.

        Parameters
        ----------
        values : dict
            Dictionary of input values

        Returns
        -------
        dict
            Updated values with defaults set
        """
        # Ensure that no_descendants and no_ascendents are lists.
        if values.get("no_descendants") is None:
            values["no_descendants"] = []
        if values.get("no_ascendents") is None:
            values["no_ascendents"] = []
        return values

    def __init__(self, **data):
        """
        Initialize the causal discovery model.

        Parameters
        ----------
        **data : dict
            Keyword arguments matching class attributes
        """
        super().__init__(**data)
        # Process the data and initialize internal state.
        self.data = self.data.drop(columns=["date"], errors="ignore")
        self.nodes = self.data.columns.tolist()
        self.graph = {node: [] for node in self.nodes}
        self.no_descendants = set(self.no_descendants)
        self.no_ascendents = set(self.no_ascendents)
        self.local_score_cache = {}
        self.inference_cache = {}

    def _visualize_dag(
        self,
        graph: dict | None = None,
        output_filename: str = "output_dag",
        output_format: str = "png",
        view: bool = False,
    ) -> Digraph:
        """
        Create visualization of a DAG using Graphviz.

        Parameters
        ----------
        graph : dict, optional
            Graph structure to visualize, by default None
        output_filename : str, optional
            Output file name, by default 'output_dag'
        output_format : str, optional
            Output file format, by default 'png'
        view : bool, optional
            Whether to display the graph, by default False

        Returns
        -------
        Digraph
            Graphviz visualization object
        """
        if graph is None:
            graph = self.graph

        dot = Digraph(comment="DAG from Bayesian Causal Discovery")
        for node in graph.keys():
            dot.node(node, node)
        for child, parents in graph.items():
            for parent in parents:
                dot.edge(parent, child)
        dot.render(output_filename, format=output_format, view=view)
        return dot

    def visualize(
        self,
        output_filename: str = "output_dag",
        output_format: str = "png",
        view: bool = False,
    ) -> Digraph:
        """
        Create visualization of the current graph structure.

        Parameters
        ----------
        output_filename : str, optional
            Output file name, by default 'output_dag'
        output_format : str, optional
            Output file format, by default 'png'
        view : bool, optional
            Whether to display the graph, by default False

        Returns
        -------
        Digraph
            Graphviz visualization object
        """
        return self._visualize_dag(self.graph, output_filename, output_format, view)

    def _is_dag(self, graph: dict) -> bool:
        """
        Check if a given graph is a Directed Acyclic Graph (DAG).

        Parameters
        ----------
        graph : dict
            Graph structure to check

        Returns
        -------
        bool
            True if graph is a DAG, False otherwise
        """
        G = nx.DiGraph()
        G.add_nodes_from(graph.keys())
        for child, parents in graph.items():
            for parent in parents:
                G.add_edge(parent, child)
        return nx.is_directed_acyclic_graph(G)

    def _get_neighbors(self, graph: dict, node: str) -> set:
        """
        Get all neighbors of a given node.

        Parameters
        ----------
        graph : dict
            Graph structure to analyze
        node : str
            Node to find neighbors for

        Returns
        -------
        set
            Set of neighboring nodes
        """
        parents = set(graph[node])
        children = {n for n, ps in graph.items() if node in ps}
        return parents.union(children)

    def _dag_to_cpdag(self, graph: dict, threshold: float = 0.1) -> dict:
        """
        Convert a DAG to a CPDAG by removing weakly supported edge orientations.

        Parameters
        ----------
        graph : dict
            Input DAG structure
        threshold : float, optional
            Score difference threshold for weak orientation, by default 0.1

        Returns
        -------
        dict
            Resulting CPDAG structure
        """
        cpdag = copy.deepcopy(graph)
        for child, parents in graph.items():
            for parent in parents:
                score_with_edge = self._compute_local_score(
                    child, list(set(graph[child]))
                )
                candidate_parents = list(set(graph[child]) - {parent})
                score_without_edge = self._compute_local_score(child, candidate_parents)
                score_diff = abs(score_with_edge - score_without_edge)
                if score_diff < threshold:
                    if parent in cpdag[child]:
                        cpdag[child].remove(parent)
        return cpdag

    def _build_model(self, node: str, parents: list) -> pm.Model:
        """
        Build a PyMC model for a given node with its parents.

        Parameters
        ----------
        node : str
            Target variable
        parents : list
            List of parent nodes

        Returns
        -------
        pm.Model
            PyMC model object
        """
        model = pm.Model()
        with model:
            if parents:
                intercept = pm.Normal(f"intercept_{node}", mu=0, sigma=1)
                betas = pm.Normal(f"beta_{node}", mu=0, sigma=1, shape=len(parents))
                parent_data = self.data[parents].values
                mu_node = intercept + pt.dot(parent_data, betas)
            else:
                intercept = pm.Normal(f"intercept_{node}", mu=0, sigma=1)
                mu_node = intercept

            sigma = pm.HalfNormal(f"sigma_{node}", sigma=1)
            pm.Normal(
                f"obs_{node}", mu=mu_node, sigma=sigma, observed=self.data[node].values
            )
        return model

    def _sample(self, model: pm.Model, method: str = "advi") -> az.InferenceData:
        """
        Sample from the PyMC model.

        Parameters
        ----------
        model : pm.Model
            PyMC model to sample from
        method : str, optional
            Sampling method ('advi' or 'mcmc'), by default 'advi'

        Returns
        -------
        az.InferenceData
            Sampling results
        """
        with model:
            if method == "advi":
                # ADVI defaults include keys for iterations and sample size.
                advi_defaults = {
                    "n": 10_000,
                    "sample_size": 2_000,
                    "progressbar": False,
                    "random_seed": 42,
                }
                # Merge instance-level sample kwargs (provided at init or run)
                advi_defaults.update(self.sample_kwargs)
                # Extract the specific ADVI parameters
                n = advi_defaults["n"]
                sample_size = advi_defaults["sample_size"]
                # pop the keys from the dict
                advi_defaults.pop("n")
                advi_defaults.pop("sample_size")
                # run
                approx = pm.fit(n=n, method="advi", **advi_defaults)
                idata = approx.sample(
                    sample_size, random_seed=advi_defaults["random_seed"]
                )
            else:
                mcmc_defaults = {
                    "tune": 1000,
                    "draws": 300,
                    "chains": 3,
                    "random_seed": 42,
                    "target_accept": 0.85,
                    "progressbar": False,
                }
                mcmc_defaults.update(self.sample_kwargs)
                idata = pm.sample(**mcmc_defaults)
        return idata

    @abstractmethod
    def _compute_local_score(self, node: str, parents: list) -> float:
        """
        Compute the local score for a node given its parents.

        Parameters
        ----------
        node : str
            Target variable
        parents : list
            List of parent nodes

        Returns
        -------
        float
            Local score value
        """
        raise NotImplementedError

    @abstractmethod
    def _compute_total_score(self, graph: dict) -> float:
        """
        Compute the total score for the entire graph.

        Parameters
        ----------
        graph : dict
            Graph structure to score

        Returns
        -------
        float
            Total score value
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, verbose: bool = False) -> tuple[dict, float]:
        """
        Run the causal discovery algorithm.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress, by default False

        Returns
        -------
        tuple[dict, float]
            Final graph structure and total score
        """
        raise NotImplementedError


class BayesianGreedySearch(BayesianCausalDiscoveryBase):
    """
    Implement Bayesian Causal Discovery using a Greedy Search approach.

    This class implements scoring functions and modular forward/backward phases
    for causal structure learning.

    Parameters
    ----------
    score_metric : str, optional
        Scoring method to use ('plls' or 'bic'), by default 'plls'

    Notes
    -----
    PLLS stands for Penalized Log-Likelihood Score
    BIC stands for Bayesian Information Criterion
    """

    score_metric: str = "plls"  # 'plls' (Penalized Log-Likelihood Score) or 'bic'

    def _compute_plls_score(
        self, node: str, parents: list, node_log_lik: float
    ) -> float:
        """
        Compute the Penalized Log-Likelihood Score (PLLS).

        Parameters
        ----------
        node : str
            Target variable
        parents : list
            List of parent nodes
        node_log_lik : float
            Log likelihood value

        Returns
        -------
        float
            PLLS score
        """
        return node_log_lik - self.penalty_weight * len(parents)

    def _compute_bic_score(
        self, node: str, parents: list, node_log_lik: float
    ) -> float:
        """
        Compute the BIC-based score.

        Parameters
        ----------
        node : str
            Target variable
        parents : list
            List of parent nodes
        node_log_lik : float
            Log likelihood value

        Returns
        -------
        float
            BIC score
        """
        n = self.data.shape[0]
        k = 1 + len(parents)
        return 2 * node_log_lik - k * np.log(n)

    def _compute_local_score(self, node: str, parents: list) -> float:
        """
        Compute the local score for a node given its parents.

        Parameters
        ----------
        node : str
            Target variable
        parents : list
            List of parent nodes

        Returns
        -------
        float
            Local score value
        """
        key = (node, tuple(sorted(parents)))
        if key in self.local_score_cache:
            return self.local_score_cache[key]

        model = self._build_model(node, parents)
        if key in self.inference_cache:
            idata = self.inference_cache[key]
        else:
            idata = self._sample(model, method="advi")
            self.inference_cache[key] = idata

        pm.compute_log_likelihood(idata, model=model, progressbar=False)
        node_log_lik = idata.log_likelihood[f"obs_{node}"].sum().item()

        if self.score_metric.lower() == "bic":
            score = self._compute_bic_score(node, parents, node_log_lik)
        else:
            score = self._compute_plls_score(node, parents, node_log_lik)

        self.local_score_cache[key] = score
        return score

    def _compute_total_score(self, graph: dict) -> float:
        """
        Compute the total score for the entire graph.

        Parameters
        ----------
        graph : dict
            Graph structure to score

        Returns
        -------
        float
            Total score value
        """
        total = 0.0
        for node, parents in graph.items():
            total += self._compute_local_score(node, parents)
        return total

    def forward_phase(self, current_score: float, verbose: bool = False) -> float:
        """
        Execute the forward phase of the greedy search.

        Parameters
        ----------
        current_score : float
            Current total score
        verbose : bool, optional
            Whether to print progress, by default False

        Returns
        -------
        float
            Updated total score
        """
        improvement = True
        while improvement:
            improvement = False
            best_delta = -np.inf
            best_move = None  # (parent, child, candidate_parents)

            for parent in self.nodes:
                if parent in self.no_descendants:  # type: ignore
                    continue
                for child in self.nodes:
                    if child in self.no_ascendents:  # type: ignore
                        continue
                    if parent == child or parent in self.graph[child]:
                        continue

                    current_score_child = self._compute_local_score(
                        child, self.graph[child]
                    )
                    best_delta_for_move = -np.inf
                    best_candidate_parents = None

                    Tj = self._get_neighbors(self.graph, child)
                    Ti = self._get_neighbors(self.graph, parent)
                    candidate_neighbors = list(
                        Tj.difference(Ti).difference({child, parent})
                    )

                    max_subset_size = 2
                    for r in range(
                        0, min(len(candidate_neighbors), max_subset_size) + 1
                    ):
                        for subset in itertools.combinations(candidate_neighbors, r):
                            candidate_parents = list(
                                set(self.graph[child]).union({parent}).union(subset)
                            )
                            candidate_graph = copy.deepcopy(self.graph)
                            candidate_graph[child] = candidate_parents
                            if not self._is_dag(candidate_graph):
                                continue
                            new_score_child = self._compute_local_score(
                                child, candidate_parents
                            )
                            delta = new_score_child - current_score_child
                            if delta > best_delta_for_move:
                                best_delta_for_move = delta
                                best_candidate_parents = candidate_parents

                    if best_delta_for_move > best_delta:
                        best_delta = best_delta_for_move
                        best_move = (parent, child, best_candidate_parents)

            if best_move is not None and best_delta > 0:
                parent, child, candidate_parents = best_move  # type: ignore
                self.graph[child] = candidate_parents
                self.graph = self._dag_to_cpdag(self.graph)
                current_score += best_delta
                if verbose:
                    print(
                        f"Forward phase: Updated parents of {child} to {candidate_parents}, "
                        f"delta = {best_delta:.2f}, new score = {current_score:.2f}"
                    )
                improvement = True
            else:
                if verbose:
                    print("Forward phase: No further improvements.")
                break
        return current_score

    def backward_phase(self, current_score: float, verbose: bool = False) -> float:
        """
        Execute the backward phase of the greedy search.

        Parameters
        ----------
        current_score : float
            Current total score
        verbose : bool, optional
            Whether to print progress, by default False

        Returns
        -------
        float
            Updated total score
        """
        improvement = True
        while improvement:
            improvement = False
            best_delta = -np.inf
            best_move = None  # (child, candidate_parents)

            for child in self.nodes:
                if child in self.no_ascendents:  # type: ignore
                    current_parents = self.graph[child]
                    if current_parents:
                        current_score_child = self._compute_local_score(
                            child, current_parents
                        )
                        candidate_parents = []  # type: ignore
                        candidate_graph = copy.deepcopy(self.graph)
                        candidate_graph[child] = candidate_parents
                        if self._is_dag(candidate_graph):
                            new_score_child = self._compute_local_score(
                                child, candidate_parents
                            )
                            delta = new_score_child - current_score_child
                            if delta > best_delta:
                                best_delta = delta
                                best_move = (child, candidate_parents)
                    continue

                current_parents = self.graph[child]
                if not current_parents:
                    continue
                current_score_child = self._compute_local_score(child, current_parents)

                for r in range(len(current_parents)):
                    for subset in itertools.combinations(current_parents, r):
                        candidate_parents = list(subset)
                        candidate_graph = copy.deepcopy(self.graph)
                        candidate_graph[child] = candidate_parents
                        if not self._is_dag(candidate_graph):
                            continue
                        new_score_child = self._compute_local_score(
                            child, candidate_parents
                        )
                        delta = new_score_child - current_score_child
                        if delta > best_delta:
                            best_delta = delta
                            best_move = (child, candidate_parents)

            if best_move is not None and best_delta > 0:
                child, candidate_parents = best_move
                self.graph[child] = candidate_parents
                self.graph = self._dag_to_cpdag(self.graph)
                current_score += best_delta
                if verbose:
                    print(
                        f"Backward phase: Updated parents of {child} to {candidate_parents}, "
                        f"delta = {best_delta:.2f}, new score = {current_score:.2f}"
                    )
                improvement = True
            else:
                if verbose:
                    print("Backward phase: No further improvements.")
                break
        return current_score

    def run(self, verbose: bool = False) -> tuple[dict, float]:
        """
        Run the Greedy Search algorithm.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print progress, by default False

        Returns
        -------
        tuple[dict, float]
            Final graph structure and total score
        """
        current_score = self._compute_total_score(self.graph)
        if verbose:
            print(f"Initial total log likelihood score: {current_score:.2f}")

        current_score = self.forward_phase(current_score, verbose=verbose)
        current_score = self.backward_phase(current_score, verbose=verbose)

        if verbose:
            print("\nFinal graph structure:")
            for node, parents in self.graph.items():
                print(f"  {node}: parents = {parents}")
            print(f"Final total log likelihood score: {current_score:.2f}")
        return self.graph, current_score
