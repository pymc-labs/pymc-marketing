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
"""Causal module."""

import re
import warnings

import networkx as nx
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from pydantic import Field, InstanceOf, validate_call
from pymc_extras.prior import Prior

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


class BuildModelFromDAG:
    """Build a PyMC probabilistic model directly from a Causal DAG and a tabular dataset.

    The class interprets a Directed Acyclic Graph (DAG) where each node is a column
    in the provided `df`. For every edge ``A -> B`` it creates a slope prior for
    the contribution of ``A`` into the mean of ``B``. Each node receives a
    likelihood prior. Dims and coords are used to align and index observed data
    via ``pm.Data`` and xarray.

    Parameters
    ----------
    dag : str
        DAG in DOT format (e.g. ``digraph { A -> B; B -> C; }``) or as a simple
        comma/newline separated list of edges (e.g. ``"A->B, B->C"``).
    df : pandas.DataFrame
        DataFrame that contains a column for every node present in the DAG and
        all columns named by the provided ``dims``.
    target : str
        Name of the target node present in both the DAG and ``df``. This is not
        used to restrict modeling but is validated to exist in the DAG.
    dims : tuple[str, ...]
        Dims for the observed variables and likelihoods (e.g. ``("date", "channel")``).
    coords : dict
        Mapping from dim names to coordinate values. All coord keys must exist as
        columns in ``df`` and will be used to pivot the data to match dims.
    model_config : dict, optional
        Optional configuration with priors for keys ``"slope"`` and ``"likelihood"``.
        Values should be ``pymc_extras.prior.Prior`` instances. Missing keys fall
        back to :pyattr:`default_model_config`.

    Examples
    --------
    Minimal example using DOT format:

    .. code-block:: python

        import pandas as pd
        import numpy as np
        from pymc_marketing.mmm.causal import BuildModelFromDAG

        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "X": np.random.normal(size=5),
                "Y": np.random.normal(size=5),
            }
        )

        dag = "digraph { X -> Y; }"
        dims = ("date",)
        coords = {"date": dates}

        builder = BuildModelFromDAG(
            dag=dag, df=df, target="Y", dims=dims, coords=coords
        )
        model = builder.build()

    Edge-list format and custom likelihood prior:

    .. code-block:: python

        from pymc_extras.prior import Prior

        dag = "X->Y"  # equivalent to the DOT example above
        model_config = {
            "likelihood": Prior(
                "StudentT", nu=5, sigma=Prior("HalfNormal", sigma=1), dims=("date",)
            ),
        }

        builder = BuildModelFromDAG(
            dag=dag,
            df=df,
            target="Y",
            dims=("date",),
            coords={"date": dates},
            model_config=model_config,
        )
        model = builder.build()
    """

    @validate_call
    def __init__(
        self,
        *,
        dag: str = Field(..., description="DAG in DOT string format or A->B list"),
        df: InstanceOf[pd.DataFrame] = Field(
            ..., description="DataFrame containing all DAG node columns"
        ),
        target: str = Field(..., description="Target node name present in DAG and df"),
        dims: tuple[str, ...] = Field(
            ..., description="Dims for observed/likelihood variables"
        ),
        coords: dict = Field(
            ...,
            description=(
                "Required coords mapping for dims and priors. All coord keys must exist as columns in df."
            ),
        ),
        model_config: dict | None = Field(
            None,
            description=(
                "Optional model config with Priors for 'slope' and 'likelihood'. "
                "Keys not supplied fall back to defaults."
            ),
        ),
    ) -> None:
        self.dag = dag
        self.df = df
        self.target = target
        self.dims = dims
        self.coords = coords

        # Parse graph and validate target
        self.graph = self._parse_dag(self.dag)
        self.nodes = list(nx.topological_sort(self.graph))
        if self.target not in self.nodes:
            raise ValueError(f"Target '{self.target}' not in DAG nodes: {self.nodes}")

        # Merge provided model_config with defaults
        provided = model_config
        self.model_config = self.default_model_config
        if provided is not None:
            self.model_config.update(provided)

        # Validate coords are present and consistent with dims, priors, and df
        self._validate_coords_required_are_consistent()

        # Validate prior dims consistency early (does not require building the model)
        self._warning_if_slope_dims_dont_match_likelihood_dims()

    @property
    def default_model_config(self) -> dict:
        """Default priors for slopes and likelihood using ``pymc_extras.Prior``.

        Returns
        -------
        dict
            Dictionary with keys ``"slope"`` and ``"likelihood"`` mapping to
            ``Prior`` instances with dims derived from :pyattr:`dims`.
        """
        slope_dims = tuple(dim for dim in (self.dims or ()) if dim != "date")
        return {
            "slope": Prior("Normal", mu=0, sigma=1, dims=slope_dims),
            "likelihood": Prior(
                "Normal",
                sigma=Prior("HalfNormal", sigma=1),
                dims=self.dims,
            ),
        }

    @staticmethod
    def _parse_dag(dag_str: str) -> nx.DiGraph:
        """Parse DOT digraph or edge-list string into a directed acyclic graph."""
        # Primary format: DOT digraph
        s = dag_str.strip()
        g = nx.DiGraph()

        if s.lower().startswith("digraph"):
            # Extract content within the first top-level {...}
            brace_start = s.find("{")
            brace_end = s.rfind("}")
            if brace_start == -1 or brace_end == -1 or brace_end <= brace_start:
                raise ValueError("Malformed DOT digraph: missing braces")
            body = s[brace_start + 1 : brace_end]

            # Remove comments (// ... or # ... at line end)
            lines = []
            for raw_line in body.splitlines():
                line = re.split(r"//|#", raw_line, maxsplit=1)[0].strip()
                if line:
                    lines.append(line)
            body = "\n".join(lines)

            # Find edges "A -> B" possibly ending with ';'
            for m in re.finditer(
                r"\b([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)\s*;?", body
            ):
                a, b = m.group(1), m.group(2)
                g.add_edge(a, b)

            # Find standalone node declarations (lines with single identifier, optional ';')
            for raw_line in body.splitlines():
                line = raw_line.strip().rstrip(";")
                if not line or "->" in line or "[" in line or "]" in line:
                    continue
                mnode = re.match(r"^([A-Za-z0-9_]+)$", line)
                if mnode:
                    g.add_node(mnode.group(1))

        else:
            # Fallback: simple comma/newline-separated "A->B" tokens
            edges: list[tuple[str, str]] = []
            for token in re.split(r"[,\n]+", s):
                token = token.strip().rstrip(";")
                if not token:
                    continue
                medge = re.match(r"^([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)$", token)
                if not medge:
                    raise ValueError(f"Invalid edge token: '{token}'")
                a, b = medge.group(1), medge.group(2)
                edges.append((a, b))
            g.add_edges_from(edges)

        if not nx.is_directed_acyclic_graph(g):
            raise ValueError("Provided graph is not a DAG.")
        return g

    def _warning_if_slope_dims_dont_match_likelihood_dims(self) -> None:
        """Warn if slope prior dims differ from likelihood dims without the 'date' dim."""
        slope_prior = self.model_config["slope"]
        likelihood_prior = self.model_config["likelihood"]

        like_dims = getattr(likelihood_prior, "dims", None)
        if isinstance(like_dims, str):
            like_dims = (like_dims,)
        elif isinstance(like_dims, list):
            like_dims = tuple(like_dims)

        # Guard against None dims (treat as empty)
        if like_dims is None:
            expected_slope_dims = ()
        else:
            expected_slope_dims = tuple(dim for dim in like_dims if dim != "date")

        slope_dims = getattr(slope_prior, "dims", None)
        if slope_dims is None or not isinstance(slope_dims, tuple):
            slope_dims = ()
        elif isinstance(slope_dims, str):
            slope_dims = (slope_dims,)
        elif isinstance(slope_dims, list):
            slope_dims = tuple(slope_dims)

        if slope_dims != expected_slope_dims:
            warnings.warn(
                (
                    "Slope prior dims "
                    f"{slope_dims if slope_dims else '()'} do not match expected dims "
                    f"{expected_slope_dims} (likelihood dims without 'date')."
                ),
                stacklevel=2,
            )

    def _validate_coords_required_are_consistent(self) -> None:
        """Validate mutual consistency among dims, coords, priors, and data columns."""
        if self.coords is None:
            raise ValueError("'coords' is required and cannot be None.")

        # 1) All coords keys must correspond to columns in the dataset
        for key in self.coords.keys():
            if key not in self.df.columns:
                raise KeyError(
                    f"Coordinate key '{key}' not found in DataFrame columns. Present columns: {list(self.df.columns)}"
                )

        # 2) Ensure dims are present in coords
        for d in self.dims:
            if d not in self.coords:
                raise ValueError(f"Missing coordinate values for dim '{d}' in coords.")

        # 3) Ensure Prior.dims exist in coords (for all top-level priors we manage)
        def _to_tuple(maybe_dims):
            if isinstance(maybe_dims, str):
                return (maybe_dims,)
            if isinstance(maybe_dims, (list, tuple)):
                return tuple(maybe_dims)
            else:
                return tuple()

        for prior_name, prior in self.model_config.items():
            if not isinstance(prior, Prior):
                continue
            for d in _to_tuple(getattr(prior, "dims", None)):
                if d not in self.coords:
                    raise ValueError(
                        f"Dim '{d}' declared in Prior '{prior_name}' must be present in coords."
                    )

        # 4) Enforce that likelihood dims match class dims exactly
        likelihood_prior = self.model_config.get("likelihood")
        if isinstance(likelihood_prior, Prior):
            likelihood_dims = _to_tuple(getattr(likelihood_prior, "dims", None))
            if likelihood_dims and tuple(self.dims) != likelihood_dims:
                raise ValueError(
                    "Likelihood Prior dims "
                    f"{likelihood_dims} must match class dims {tuple(self.dims)}. "
                    "When supplying a custom model_config, ensure likelihood.dims equals the 'dims' argument."
                )

    def _parents(self, node: str) -> list[str]:
        """Return the list of parent node names for the given DAG node."""
        return list(self.graph.predecessors(node))

    def build(self) -> pm.Model:
        """Construct and return the PyMC model implied by the DAG and data.

        The method creates a ``pm.Data`` container for every node to align the
        observed data with the declared ``dims``. For each edge ``A -> B``, a
        slope prior is instantiated from ``model_config['slope']`` and used in the
        mean of node ``B``'s likelihood, which is instantiated from
        ``model_config['likelihood']``.

        Returns
        -------
        pymc.Model
            A fully specified model with slopes and likelihoods for all nodes.

        Examples
        --------
        Build a model and sample from it:

        .. code-block:: python

            builder = BuildModelFromDAG(
                dag="A->B", df=df, target="B", dims=("date",), coords={"date": dates}
            )
            model = builder.build()
            with model:
                idata = pm.sample(100, tune=100, chains=2, cores=2)

        Multi-dimensional dims (e.g. date and country):

        .. code-block:: python

            dims = ("date", "country")
            coords = {"date": dates, "country": ["Venezuela", "Colombia"]}
            builder = BuildModelFromDAG(
                dag="A->B, B->Y", df=df, target="Y", dims=dims, coords=coords
            )
            model = builder.build()
        """
        dims = self.dims
        coords = self.coords

        with pm.Model(coords=coords) as model:
            data_containers: dict[str, pm.Data] = {}
            for node in self.nodes:
                if node not in self.df.columns:
                    raise KeyError(f"Column '{node}' not found in df.")
                # Ensure observed data has shape consistent with declared dims by pivoting via xarray
                indexed = self.df.set_index(list(dims))
                xarr = indexed.to_xarray()[node]
                values = xarr.values

                data_containers[node] = pm.Data(f"_{node}", values, dims=dims)

            # For each node add slope priors per parent and likelihood with sigma prior
            slope_rvs: dict[tuple[str, str], pt.TensorVariable] = {}

            # Create priors in a stable deterministic order
            for node in self.nodes:
                parents = self._parents(node)
                # Slopes for each parent -> node
                parents = self._parents(node)
                mu_expr = 0
                for parent in parents:
                    slope_name = f"{parent.lower()}{node.lower()}"
                    slope_rv = self.model_config["slope"].create_variable(slope_name)
                    slope_rvs[(parent, node)] = slope_rv
                    mu_expr += slope_rv * data_containers[parent]

                self.model_config["likelihood"].create_likelihood_variable(
                    name=node,
                    mu=mu_expr,
                    observed=data_containers[node],
                )

            self.model = model
        return self.model

    def model_graph(self):
        """Return a Graphviz visualization of the built PyMC model.

        Returns
        -------
        graphviz.Source
            Graphviz object representing the model graph.

        Examples
        --------
        .. code-block:: python

            model = builder.build()
            g = builder.model_graph()
            g
        """
        if not hasattr(self, "model"):
            raise RuntimeError("Call build() first.")
        return pm.model_to_graphviz(self.model)

    def dag_graph(self):
        """Return a copy of the parsed DAG as a NetworkX directed graph.

        Returns
        -------
        networkx.DiGraph
            A directed acyclic graph with the same nodes and edges as the input DAG.

        Examples
        --------
        .. code-block:: python

            g = builder.dag_graph()
            list(g.edges())
        """
        g = nx.DiGraph()
        g.add_nodes_from(self.graph.nodes)
        g.add_edges_from(self.graph.edges)
        return g


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
