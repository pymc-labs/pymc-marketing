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

from __future__ import annotations

import itertools as it
import re
import warnings
from collections.abc import Sequence
from typing import Annotated, Literal, NotRequired, TypedDict

try:
    import networkx as nx
except ImportError:  # Optional dependency
    nx = None  # type: ignore[assignment]

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
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


class TestResult(TypedDict):
    """Conditional independence test statistics recorded during fitting."""

    bic0: float
    bic1: float
    delta_bic: float
    logBF10: float
    BF10: float
    independent: bool
    conditioning_set: list[str]
    forced: NotRequired[bool]


EMPTY_CONDITION_SET: frozenset[str] = frozenset()


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
        Optional configuration with priors for keys ``"intercept"``, ``"slope"`` and
        ``"likelihood"``. Values should be ``pymc_extras.prior.Prior`` instances.
        Missing keys fall back to :pyattr:`default_model_config`.

    Examples
    --------
    Minimal example using DOT format:

    .. code-block:: python

        import numpy as np
        import pandas as pd

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
                "Optional model config with Priors for 'intercept', 'slope' and "
                "'likelihood'. Keys not supplied fall back to defaults."
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

        # Validate required priors are present and of correct type
        self._validate_model_config_priors()

        # Validate coords are present and consistent with dims, priors, and df
        self._validate_coords_required_are_consistent()

        # Validate prior dims consistency early (does not require building the model)
        self._warning_if_slope_dims_dont_match_likelihood_dims()
        self._validate_intercept_dims_match_slope_dims()

    @property
    def default_model_config(self) -> dict[str, Prior]:
        """Default priors for intercepts, slopes and likelihood using ``pymc_extras.Prior``.

        Returns
        -------
        dict
            Dictionary with keys ``"intercept"``, ``"slope"`` and ``"likelihood"``
            mapping to ``Prior`` instances with dims derived from
            :pyattr:`dims`.
        """
        slope_dims = tuple(dim for dim in (self.dims or ()) if dim != "date")
        return {
            "intercept": Prior("Normal", mu=0, sigma=2, dims=slope_dims),
            "slope": Prior("Normal", mu=0, sigma=2, dims=slope_dims),
            "likelihood": Prior(
                "Normal",
                sigma=Prior("HalfNormal", sigma=2),
                dims=self.dims,
            ),
        }

    @staticmethod
    def _parse_dag(dag_str: str) -> nx.DiGraph:
        """Parse DOT digraph or edge-list string into a directed acyclic graph."""
        if nx is None:
            raise ImportError(
                "To use Causal Graph functionality, please install the optional dependencies with: "
                "pip install pymc-marketing[dag]"
            )
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

    def _validate_intercept_dims_match_slope_dims(self) -> None:
        """Ensure intercept prior dims match slope prior dims exactly."""

        def _to_tuple(maybe_dims):
            if maybe_dims is None:
                return tuple()
            if isinstance(maybe_dims, str):
                return (maybe_dims,)
            if isinstance(maybe_dims, list | tuple):
                return tuple(maybe_dims)
            return tuple()

        slope_dims = _to_tuple(getattr(self.model_config["slope"], "dims", None))
        intercept_dims = _to_tuple(
            getattr(self.model_config["intercept"], "dims", None)
        )

        if slope_dims != intercept_dims:
            raise ValueError(
                "model_config['intercept'].dims must match model_config['slope'].dims. "
                f"Got intercept dims {intercept_dims or '()'} and slope dims {slope_dims or '()'}."
            )

    def _validate_model_config_priors(self) -> None:
        """Ensure required model_config entries are Prior instances.

        Enforces that keys 'slope' and 'likelihood' exist and are Prior objects,
        so downstream code can safely index and call Prior helper methods.
        """
        required_keys = ("intercept", "slope", "likelihood")
        for key in required_keys:
            if key not in self.model_config:
                raise ValueError(f"model_config must include '{key}' as a Prior.")
        for key in required_keys:
            if not isinstance(self.model_config[key], Prior):
                raise TypeError(
                    f"model_config['{key}'] must be a Prior, got "
                    f"{type(self.model_config[key]).__name__}."
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
            if isinstance(maybe_dims, list | tuple):
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
        likelihood_prior = self.model_config["likelihood"]
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
                mu_expr = 0
                for parent in parents:
                    slope_name = f"{parent.lower()}:{node.lower()}"
                    slope_rv = self.model_config["slope"].create_variable(slope_name)
                    slope_rvs[(parent, node)] = slope_rv
                    mu_expr += slope_rv * data_containers[parent]
                intercept_rv = self.model_config["intercept"].create_variable(
                    f"{node.lower()}_intercept"
                )

                self.model_config["likelihood"].create_likelihood_variable(
                    name=node,
                    mu=mu_expr + intercept_rv,
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
        if nx is None:
            raise ImportError(
                "To use Causal Graph functionality, please install the optional dependencies with: "
                "pip install pymc-marketing[dag]"
            )
        g = nx.DiGraph()
        g.add_nodes_from(self.graph.nodes)
        g.add_edges_from(self.graph.edges)
        return g


class TBFPC:
    r"""
    Target-first Bayes Factor PC (TBF-PC) causal discovery algorithm.

    This algorithm is a target-oriented variant of the Peter–Clark (PC) algorithm,
    using Bayes factors (via ΔBIC approximation) as the conditional independence test.

    For each conditional independence test of the form

    .. math::

        H_0 : Y \perp X \mid S
        \quad \text{vs.} \quad
        H_1 : Y \not\!\perp X \mid S

    we compare two linear models:

    .. math::

        M_0 : Y \sim S
        \\
        M_1 : Y \sim S + X

    where :math:`S` is a conditioning set of variables.

    The Bayesian Information Criterion (BIC) is defined as

    .. math::

        \mathrm{BIC}(M) = n \log\!\left(\frac{\mathrm{RSS}}{n}\right)
                          + k \log(n),

    with residual sum of squares :math:`\mathrm{RSS}`, sample size :math:`n`,
    and number of parameters :math:`k`.

    The Bayes factor is approximated by

    .. math::

        \log \mathrm{BF}_{10} \approx -\tfrac{1}{2}
        \left[ \mathrm{BIC}(M_1) - \mathrm{BIC}(M_0) \right].

    Independence is declared if :math:`\mathrm{BF}_{10} < \tau`,
    where :math:`\tau` is set via the ``bf_thresh`` parameter.

    Target Edge Rules
    -----------------
    Different rules govern how driver → target edges are retained:

    - ``"any"``:
      keep :math:`X \to Y` unless **any** conditioning set renders
      :math:`X \perp Y \mid S`.
    - ``"conservative"``:
      keep :math:`X \to Y` if **at least one** conditioning set shows
      dependence.
    - ``"fullS"``:
      test only with the **full set** of other drivers as :math:`S`.

    Examples
    --------
    **1. Basic usage with full conditioning set**

    .. code-block:: python

        import numpy as np, pandas as pd

        rng = np.random.default_rng(7)
        n = 2000
        C = rng.gamma(2,1,n)
        A = 0.7*C + rng.gamma(2,1,n)
        D = 0.5*C + rng.gamma(2,1,n)
        B = 0.8*A + rng.gamma(2,1,n)
        Y = 0.9*B + 0.6*D + 0.7*C + rng.gamma(2,1,n)

        df = pd.DataFrame({"A":A,"B":B,"C":C,"D":D,"Y":Y})
        df = (df - df.mean())/df.std()  # recommended scaling

        model = TBFPC(target="Y", target_edge_rule="fullS")
        model.fit(df, drivers=["A","B","C","D"])

        print(model.get_directed_edges())
        print(model.get_undirected_edges())
        print(model.to_digraph())

    **2. Using forbidden edges**

    You can specify edges that must *not* be tested or included
    (prior knowledge about the domain).

    .. code-block:: python

        model = TBFPC(
            target="Y",
            target_edge_rule="any",
            forbidden_edges=[("A","C")]  # forbid A--C
        )
        model.fit(df, drivers=["A","B","C","D"])
        print(model.to_digraph())

    **3. Conservative rule**

    Keeps driver → target edges if **any conditioning set**
    shows dependence.

    .. code-block:: python

        model = TBFPC(target="Y", target_edge_rule="conservative")
        model.fit(df, drivers=["A","B","C","D"])
        print(model.to_digraph())

    References
    ----------
    - Spirtes, Glymour, Scheines (2000). *Causation, Prediction, and Search*. MIT Press. [PC algorithm]
    - Spirtes & Glymour (1991). "An Algorithm for Fast Recovery of Sparse Causal Graphs."
    - Kass, R. & Raftery, A. (1995). "Bayes Factors."
    """

    @validate_call(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        target: Annotated[
            str,
            Field(
                min_length=1,
                description="Name of the outcome variable to orient the search.",
            ),
        ],
        *,
        target_edge_rule: Literal["any", "conservative", "fullS"] = "any",
        bf_thresh: Annotated[float, Field(gt=0.0)] = 1.0,
        forbidden_edges: Sequence[tuple[str, str]] | None = None,
        required_edges: Sequence[tuple[str, str]] | None = None,
    ):
        """Create a new TBFPC causal discovery model.

        Parameters
        ----------
        target
            Variable name for the model outcome; must be present in the data
            used during fitting.
        target_edge_rule
            Rule that controls which driver → target edges are retained.
            Options are ``"any"``, ``"conservative"``, and ``"fullS"``.
        bf_thresh
            Positive Bayes factor threshold applied during conditional
            independence tests.
        forbidden_edges
            Optional sequence of node pairs that must not be connected in the
            learned graph.
        required_edges
            Optional sequence of directed ``(u, v)`` pairs that must be present
            in the learned graph as ``u -> v``.
        """
        warnings.warn(
            "TBFPC is experimental and its API may change; use with caution.",
            UserWarning,
            stacklevel=2,
        )

        self.target = target
        self.target_edge_rule = target_edge_rule
        self.bf_thresh = float(bf_thresh)
        self.forbidden_edges: set[tuple[str, str]] = set(forbidden_edges or [])
        self.required_edges: set[tuple[str, str]] = set(required_edges or [])

        conflicts = [
            (u, v)
            for (u, v) in self.required_edges
            if (u, v) in self.forbidden_edges or (v, u) in self.forbidden_edges
        ]
        if conflicts:
            conflict_str = ", ".join(f"{u}->{v}" for u, v in conflicts)
            raise ValueError(
                f"Required edges conflict with forbidden edges: {conflict_str}"
            )

        conflicts = [
            (u, v)
            for (u, v) in self.required_edges
            if (u, v) in self.forbidden_edges or (v, u) in self.forbidden_edges
        ]
        if conflicts:
            conflict_str = ", ".join(f"{u}->{v}" for u, v in conflicts)
            raise ValueError(
                f"Required edges conflict with forbidden edges: {conflict_str}"
            )

        # Internal state
        self.sep_sets: dict[tuple[str, str], set[str]] = {}
        self._adj_directed: set[tuple[str, str]] = set()
        self._adj_undirected: set[tuple[str, str]] = set()
        self.nodes_: list[str] = []
        self.test_results: dict[tuple[str, str, frozenset[str]], TestResult] = {}

        # Shared response vector for symbolic BIC computation
        # Initialized with placeholder; will be updated with actual data during fitting
        self.y_sh = pytensor.shared(np.zeros(1, dtype="float64"), name="y_sh")
        self._bic_fn = self._build_symbolic_bic_fn()

    @staticmethod
    def _bitmasks(k: int):
        """Yield tuples of 0/1 of length k (fast product without importing itertools)."""
        # Equivalent to itertools.product([0,1], repeat=k) but minimal
        if k == 0:
            yield ()
            return
        stack = [0] * k
        i = 0
        while True:
            if i < k:
                stack[i] = 0
                i += 1
                continue
            yield tuple(stack)
            # increment like binary counter
            i -= 1
            while i >= 0 and stack[i] == 1:
                i -= 1
            if i < 0:
                break
            stack[i] = 1
            i += 1

    @staticmethod
    def _parse_cpdag_dot(
        dot: str,
    ) -> tuple[set[str], set[tuple[str, str]], set[tuple[str, str]]]:
        """Minimal DOT parser for a single CPDAG block."""
        import re

        digraph = re.search(r"digraph\b[^{}]*\{(.*?)\}", dot, flags=re.DOTALL)
        if not digraph:
            raise ValueError("No 'digraph { ... }' block found.")

        body = digraph.group(1)
        nodes: set[str] = set()
        directed: set[tuple[str, str]] = set()
        undirected: set[tuple[str, str]] = set()

        node_re = re.compile(r'^\s*"([^"]+)"\s*(?:\[[^\]]*\])?\s*;\s*$')
        edge_re = re.compile(r'^\s*"([^"]+)"\s*->\s*"([^"]+)"\s*(\[[^\]]*\])?\s*;\s*$')

        def is_undirected(attrs: str | None) -> bool:
            if not attrs:
                return False
            low = attrs.lower()
            return "style=dashed" in low and "dir=none" in low

        for raw in body.splitlines():
            line = raw.strip()
            if not line or line.startswith("//"):
                continue
            # node?
            m = node_re.match(line)
            if m:
                nodes.add(m.group(1))
                continue
            # edge?
            m = edge_re.match(line)
            if m:
                u, v, attrs = m.group(1), m.group(2), m.group(3)
                nodes.update((u, v))
                if is_undirected(attrs):
                    undirected.add((u, v) if u <= v else (v, u))
                else:
                    directed.add((u, v))
                continue
            # ignore other lines (global styles etc.)

        return nodes, directed, undirected

    @staticmethod
    def _is_acyclic(
        nodes: set[str], edges: list[tuple[str, str]] | set[tuple[str, str]]
    ) -> bool:
        """DFS cycle check."""
        adj: dict[str, list[str]] = {u: [] for u in nodes}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, [])  # ensure key exists
        state = {u: 0 for u in nodes}  # 0=unseen, 1=visiting, 2=done

        def dfs(u: str) -> bool:
            state[u] = 1
            for w in adj[u]:
                if state[w] == 1:
                    return False
                if state[w] == 0 and not dfs(w):
                    return False
            state[u] = 2
            return True

        return all(state[u] or dfs(u) for u in nodes)

    def _dot_from_edges(
        self, nodes: set[str], edges: list[tuple[str, str]] | set[tuple[str, str]]
    ) -> str:
        """Render a fully directed graph to DOT; highlights target if present."""
        lines = ["digraph G {", "  node [shape=ellipse];"]
        for n in sorted(nodes):
            if hasattr(self, "target") and n == self.target:
                lines.append(f'  "{n}" [style=filled, fillcolor="#eef5ff"];')
            else:
                lines.append(f'  "{n}";')
        for u, v in sorted(edges):
            lines.append(f'  "{u}" -> "{v}";')
        lines.append("}")
        return "\n".join(lines).replace("\\n'", "\\n").replace("'\\n", "\\n")  # hygiene

    def _key(self, u: str, v: str) -> tuple[str, str]:
        """Return a sorted 2-tuple key for an undirected edge between ``u`` and ``v``."""
        return (u, v) if u <= v else (v, u)

    def _set_sep(self, u: str, v: str, S: Sequence[str]) -> None:
        """Record the separation set ``S`` for the node pair ``(u, v)``."""
        self.sep_sets[self._key(u, v)] = set(S)

    def _has_forbidden(self, u: str, v: str) -> bool:
        """Return True if edge ``u—v`` is forbidden in either direction."""
        return (u, v) in self.forbidden_edges or (v, u) in self.forbidden_edges

    def _is_required(self, u: str, v: str) -> bool:
        """Return True if the directed edge ``u -> v`` is required."""
        return (u, v) in self.required_edges

    def _add_directed(self, u: str, v: str) -> None:
        """Add a directed edge ``u -> v`` if not forbidden; drop undirected if present."""
        if not self._has_forbidden(u, v):
            self._adj_undirected.discard(self._key(u, v))
            self._adj_directed.add((u, v))

    def _add_undirected(self, u: str, v: str) -> None:
        """Add an undirected edge ``u -- v`` if allowed and not already directed."""
        if (
            not self._has_forbidden(u, v)
            and (u, v) not in self._adj_directed
            and (v, u) not in self._adj_directed
            and not self._is_required(u, v)
            and not self._is_required(v, u)
        ):
            self._adj_undirected.add(self._key(u, v))

    def _remove_all(self, u: str, v: str) -> None:
        """Remove any edge (directed or undirected) between ``u`` and ``v``."""
        if self._is_required(u, v) or self._is_required(v, u):
            return
        self._adj_undirected.discard(self._key(u, v))
        self._adj_directed.discard((u, v))
        self._adj_directed.discard((v, u))

    def _enforce_required_edges(self) -> None:
        """Force required edges to appear as directed adjacencies."""
        for u, v in self.required_edges:
            self._adj_undirected.discard(self._key(u, v))
            self._adj_directed.discard((v, u))
            self._adj_directed.add((u, v))
            self.test_results[(u, v, EMPTY_CONDITION_SET)] = {
                "bic0": float("nan"),
                "bic1": float("nan"),
                "delta_bic": float("nan"),
                "logBF10": float("nan"),
                "BF10": float("nan"),
                "independent": False,
                "conditioning_set": [],
                "forced": True,
            }

    def _validate_required_nodes(self, drivers: Sequence[str]) -> None:
        """Ensure required edges reference known nodes."""
        allowed = set(drivers) | {self.target}
        missing: set[str] = set()
        for u, v in self.required_edges:
            if u not in allowed:
                missing.add(u)
            if v not in allowed:
                missing.add(v)
        if missing:
            raise ValueError(
                "Required edges reference unknown nodes: " + ", ".join(sorted(missing))
            )

    def _build_symbolic_bic_fn(self):
        """Build a BIC callable using a fast solver with a pseudoinverse fallback."""
        X = pt.matrix("X")
        n = pt.iscalar("n")

        xtx = pt.dot(X.T, X)
        xty = pt.dot(X.T, self.y_sh)

        beta_solve = pt.linalg.solve(xtx, xty)
        resid_solve = self.y_sh - pt.dot(X, beta_solve)
        rss_solve = pt.sum(resid_solve**2)

        beta_pinv = pt.nlinalg.pinv(X) @ self.y_sh
        resid_pinv = self.y_sh - pt.dot(X, beta_pinv)
        rss_pinv = pt.sum(resid_pinv**2)

        k = X.shape[1]

        nf = pt.cast(n, "float64")
        rss_solve_safe = pt.maximum(rss_solve, np.finfo("float64").tiny)
        rss_pinv_safe = pt.maximum(rss_pinv, np.finfo("float64").tiny)

        bic_solve = nf * pt.log(rss_solve_safe / nf) + k * pt.log(nf)
        bic_pinv = nf * pt.log(rss_pinv_safe / nf) + k * pt.log(nf)

        bic_solve_fn = pytensor.function(
            [X, n], [bic_solve, rss_solve], on_unused_input="ignore", mode="FAST_RUN"
        )
        bic_pinv_fn = pytensor.function(
            [X, n], bic_pinv, on_unused_input="ignore", mode="FAST_RUN"
        )

        def bic_fn(X_val: np.ndarray, n_val: int) -> float:
            try:
                bic_value, rss_value = bic_solve_fn(X_val, n_val)
                if np.isfinite(rss_value) and rss_value > np.finfo("float64").tiny:
                    return float(bic_value)
            except (np.linalg.LinAlgError, RuntimeError, ValueError):
                pass
            return float(bic_pinv_fn(X_val, n_val))

        return bic_fn

    def _ci_independent(
        self, df: pd.DataFrame, x: str, y: str, cond: Sequence[str]
    ) -> bool:
        """Return True if ΔBIC indicates independence of ``x`` and ``y`` given ``cond``."""
        if self._has_forbidden(x, y):
            return True
        if self._is_required(x, y) or self._is_required(y, x):
            self.test_results[(x, y, frozenset(cond))] = TestResult(
                bic0=float("nan"),
                bic1=float("nan"),
                delta_bic=float("nan"),
                logBF10=float("nan"),
                BF10=float("nan"),
                independent=False,
                conditioning_set=list(cond),
                forced=True,
            )
            return False

        n = len(df)
        self.y_sh.set_value(df[y].to_numpy().astype("float64"))

        if len(cond) == 0:
            X0 = np.ones((n, 1))
        else:
            X0 = np.column_stack([np.ones(n), df[list(cond)].to_numpy()])
        X1 = np.column_stack([X0, df[x].to_numpy()])

        bic0 = float(self._bic_fn(X0, n))
        bic1 = float(self._bic_fn(X1, n))

        delta_bic = bic1 - bic0
        logBF10 = -0.5 * delta_bic
        BF10 = np.exp(logBF10)

        independence = BF10 < self.bf_thresh
        result: TestResult = {
            "bic0": bic0,
            "bic1": bic1,
            "delta_bic": delta_bic,
            "logBF10": logBF10,
            "BF10": BF10,
            "independent": independence,
            "conditioning_set": list(cond),
        }
        self.test_results[(x, y, frozenset(cond))] = result

        return independence

    def _test_target_edges(self, df: pd.DataFrame, drivers: Sequence[str]) -> None:
        """Phase 1: test driver→target edges according to ``target_edge_rule``."""
        for xi in drivers:
            neighbor_sets = [d for d in drivers if d != xi]
            max_k = min(3, len(neighbor_sets))
            all_sets = [
                tuple(S)
                for k in range(max_k + 1)
                for S in it.combinations(neighbor_sets, k)
            ]

            if self.target_edge_rule == "any":
                keep = True
                for S in all_sets:
                    if self._ci_independent(df, xi, self.target, S):
                        self._set_sep(xi, self.target, S)
                        keep = False
                        break
                if keep:
                    self._add_directed(xi, self.target)
                else:
                    self._remove_all(xi, self.target)

            elif self.target_edge_rule == "conservative":
                indep_all = True
                for S in all_sets:
                    if not self._ci_independent(df, xi, self.target, S):
                        indep_all = False
                    else:
                        self._set_sep(xi, self.target, S)
                if indep_all:
                    self._remove_all(xi, self.target)
                else:
                    self._add_directed(xi, self.target)

            elif self.target_edge_rule == "fullS":
                S = tuple(neighbor_sets)
                if self._ci_independent(df, xi, self.target, S):
                    self._set_sep(xi, self.target, S)
                    self._remove_all(xi, self.target)
                else:
                    self._add_directed(xi, self.target)

    def _test_driver_skeleton(self, df: pd.DataFrame, drivers: Sequence[str]) -> None:
        """Phase 2: build the undirected driver skeleton via pairwise CI tests."""
        for xi, xj in it.combinations(drivers, 2):
            others = [d for d in drivers if d not in (xi, xj)]
            max_k = min(3, len(others))
            dependent = True
            sep_rec = False
            for k in range(max_k + 1):
                for S in it.combinations(others, k):
                    if self._ci_independent(df, xi, xj, S):
                        self._set_sep(xi, xj, S)
                        dependent = False
                        sep_rec = True
                        break
                if sep_rec:
                    break
            if dependent:
                self._add_undirected(xi, xj)
            else:
                self._remove_all(xi, xj)

    def fit(self, df: pd.DataFrame, drivers: Sequence[str]):
        """Fit the TBFPC procedure to the supplied dataframe.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataset containing the target column and every candidate driver.
        drivers : Sequence[str]
            Iterable of column names to treat as potential drivers of the
            target.

        Returns
        -------
        TBFPC
            The fitted instance (``self``) with internal adjacency structures
            populated.

        Examples
        --------
        .. code-block:: python

            model = TBFPC(target="Y", target_edge_rule="fullS")
            model.fit(df, drivers=["A", "B", "C"])
        """
        self._validate_required_nodes(drivers)

        self.sep_sets.clear()
        self._adj_directed.clear()
        self._adj_undirected.clear()
        self.test_results.clear()

        self._enforce_required_edges()
        self._test_target_edges(df, drivers)
        self._test_driver_skeleton(df, drivers)
        self._enforce_required_edges()

        self.nodes_ = [*list(drivers), self.target]
        return self

    def get_directed_edges(self) -> list[tuple[str, str]]:
        """Return directed edges learned by the algorithm.

        Returns
        -------
        list[tuple[str, str]]
            Sorted list of ``(u, v)`` pairs representing oriented edges.

        Examples
        --------
        .. code-block:: python

            directed = model.get_directed_edges()
        """
        return sorted(self._adj_directed)

    def get_undirected_edges(self) -> list[tuple[str, str]]:
        """Return undirected edges remaining after orientation.

        Returns
        -------
        list[tuple[str, str]]
            Sorted list of ``(u, v)`` pairs for unresolved adjacencies.

        Examples
        --------
        .. code-block:: python

            skeleton = model.get_undirected_edges()
        """
        return sorted(self._adj_undirected)

    def get_test_results(self, x: str, y: str) -> list[TestResult]:
        """Return ΔBIC diagnostics for the unordered pair ``(x, y)``.

        Parameters
        ----------
        x : str
            Name of the first variable in the pair.
        y : str
            Name of the second variable in the pair.

        Returns
        -------
        list[dict[str, float]]
            Each dictionary contains ``bic0``, ``bic1``, ``delta_bic``,
            ``logBF10``, ``BF10``, and the conditioning set used during the
            test.

        Examples
        --------
        .. code-block:: python

            stats = model.get_test_results("A", "Y")
        """
        return [v for (xi, yi, _), v in self.test_results.items() if {xi, yi} == {x, y}]

    def summary(self) -> str:
        """Render a text summary of the learned graph and test count.

        Returns
        -------
        str
            Multiline string describing directed edges, undirected edges, and
            the number of conditional independence tests executed.

        Examples
        --------
        .. code-block:: python

            print(model.summary())
        """
        lines = ["=== Directed edges ==="]
        for u, v in self.get_directed_edges():
            suffix = " [required]" if self._is_required(u, v) else ""
            lines.append(f"{u} -> {v}{suffix}")
        lines.append("=== Undirected edges ===")
        for u, v in self.get_undirected_edges():
            lines.append(f"{u} -- {v}")
        lines.append("=== Number of CI tests run ===")
        lines.append(str(len(self.test_results)))
        return "\n".join(lines)

    def to_digraph(self) -> str:
        """Return the learned graph encoded in DOT format.

        Returns
        -------
        str
            DOT string compatible with Graphviz rendering utilities.

        Examples
        --------
        .. code-block:: python

            dot_str = model.to_digraph()
        """
        lines = ["digraph G {", "  node [shape=ellipse];"]
        for n in self.nodes_:
            if n == self.target:
                lines.append(f'  "{n}" [style=filled, fillcolor="#eef5ff"];')
            else:
                lines.append(f'  "{n}";')
        for u, v in self.get_directed_edges():
            attrs = " [color=darkgreen, penwidth=2]" if self._is_required(u, v) else ""
            lines.append(f'  "{u}" -> "{v}"{attrs};')
        for u, v in self.get_undirected_edges():
            lines.append(f'  "{u}" -> "{v}" [style=dashed, dir=none];')
        lines.append("}")
        return "\n".join(lines)

    def get_all_cdags_from_cpdag(self, dot_cpdag: str | None = None) -> list[str]:
        """
        Enumerate all acyclic orientations (consistent extensions) of the CPDAG.

        Parameters
        ----------
        dot_cpdag : str | None
            If provided, parse the CPDAG from this DOT string (expects undirected
            edges encoded as `[style=dashed, dir=none]`). If None, use the model's
            current CPDAG from `self.get_directed_edges()` and `self.get_undirected_edges()`.

        Returns
        -------
        list[str]
            A list of DOT strings, each representing a fully oriented DAG (no dashed edges).
        """
        nodes, fixed_dir, undirected = (
            self._parse_cpdag_dot(dot_cpdag)
            if dot_cpdag is not None
            else (
                set(self.nodes_),
                set(self.get_directed_edges()),
                set(self.get_undirected_edges()),
            )
        )

        if not undirected:
            # Already a DAG: validate acyclicity and return it
            edges = sorted(fixed_dir)
            if self._is_acyclic(nodes, edges):
                return [self._dot_from_edges(nodes, edges)]
            return []

        cdags: list[str] = []
        und = sorted({self._key(u, v) for (u, v) in undirected})  # canonical ordering
        for mask in self._bitmasks(len(und)):
            oriented = list(fixed_dir)
            # bit 0 -> u->v, bit 1 -> v->u
            oriented.extend(
                (u, v) if b == 0 else (v, u)
                for b, (u, v) in zip(mask, und, strict=False)
            )
            if self._is_acyclic(nodes, oriented):
                cdags.append(self._dot_from_edges(nodes, oriented))
        return cdags


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
    ) -> CausalGraphModel:
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
