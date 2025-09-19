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

import itertools as it
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as tt

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

    def __init__(
        self,
        target: str,
        *,
        target_edge_rule: str = "any",
        bf_thresh: float = 1.0,
        forbidden_edges: Sequence[tuple[str, str]] | None = None,
    ):
        warnings.warn(
            "TBFPC is experimental and its API may change; use with caution.",
            UserWarning,
            stacklevel=2,
        )
        if not isinstance(target, str) or not target:
            raise ValueError("target must be a non-empty string")
        allowed_rules = {"any", "conservative", "fullS"}
        if target_edge_rule not in allowed_rules:
            raise ValueError(f"target_edge_rule must be one of {allowed_rules}")
        if not isinstance(bf_thresh, (int, float)) or bf_thresh <= 0:
            raise ValueError("bf_thresh must be a positive float")

        self.target = target
        self.target_edge_rule = target_edge_rule
        self.bf_thresh = float(bf_thresh)
        self.forbidden_edges: set[tuple[str, str]] = set(forbidden_edges or [])

        # Internal state
        self.sep_sets: dict[tuple[str, str], set[str]] = {}
        self._adj_directed: set[tuple[str, str]] = set()
        self._adj_undirected: set[tuple[str, str]] = set()
        self.nodes_: list[str] = []
        self.test_results: dict[tuple[str, str, frozenset], dict[str, float]] = {}

        # Shared response vector for symbolic BIC
        self.y_sh = pytensor.shared(np.zeros(1, dtype="float64"), name="y_sh")
        self._bic_fn = self._build_symbolic_bic_fn()

    # ---------------------------------------------------------------------
    # Graph utils
    # ---------------------------------------------------------------------
    def _key(self, u: str, v: str) -> tuple[str, str]:
        """Return a sorted 2-tuple key for an undirected edge between ``u`` and ``v``."""
        return (u, v) if u <= v else (v, u)

    def _set_sep(self, u: str, v: str, S: Sequence[str]) -> None:
        """Record the separation set ``S`` for the node pair ``(u, v)``."""
        self.sep_sets[self._key(u, v)] = set(S)

    def _has_forbidden(self, u: str, v: str) -> bool:
        """Return True if edge ``u—v`` is forbidden in either direction."""
        return (u, v) in self.forbidden_edges or (v, u) in self.forbidden_edges

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
        ):
            self._adj_undirected.add(self._key(u, v))

    def _remove_all(self, u: str, v: str) -> None:
        """Remove any edge (directed or undirected) between ``u`` and ``v``."""
        self._adj_undirected.discard(self._key(u, v))
        self._adj_directed.discard((u, v))
        self._adj_directed.discard((v, u))

    # ---------------------------------------------------------------------
    # Statistical methods
    # ---------------------------------------------------------------------
    def _build_symbolic_bic_fn(self):
        """Build and compile a function to compute BIC given a design matrix ``X`` and sample size ``n``."""
        X = tt.matrix("X")
        n = tt.iscalar("n")

        beta = tt.nlinalg.pinv(X) @ self.y_sh
        resid = self.y_sh - X @ beta
        rss = tt.sum(resid**2)
        k = X.shape[1]

        bic = n * tt.log(rss / n) + k * tt.log(n)
        return pytensor.function([X, n], bic)

    def _ci_independent(
        self, df: pd.DataFrame, x: str, y: str, cond: Sequence[str]
    ) -> bool:
        """Return True if ΔBIC indicates independence of ``x`` and ``y`` given ``cond``."""
        if self._has_forbidden(x, y):
            return True

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

        result = {
            "bic0": bic0,
            "bic1": bic1,
            "delta_bic": delta_bic,
            "logBF10": logBF10,
            "BF10": BF10,
            "independent": BF10 < self.bf_thresh,
            "conditioning_set": list(cond),
        }
        self.test_results[(x, y, frozenset(cond))] = result

        return result["independent"]

    # ---------------------------------------------------------------------
    # Algorithm phases
    # ---------------------------------------------------------------------
    def _test_target_edges(self, df: pd.DataFrame, drivers: Sequence[str]) -> None:
        """Phase 1: test driver→target edges according to ``target_edge_rule``."""
        for xi in drivers:
            nbrs = [d for d in drivers if d != xi]
            max_k = min(3, len(nbrs))
            all_sets = [S for k in range(max_k + 1) for S in it.combinations(nbrs, k)]

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
                S = tuple(nbrs)
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

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, drivers: Sequence[str]):
        """Run the two-phase causal discovery procedure on ``df``.

        This learns directed edges into the target and an undirected driver skeleton.

        .. code-block:: python

            model = TBFPC(target="Y", target_edge_rule="fullS")
            model.fit(df, drivers=["A", "B", "C"])  # returns self
        """
        self.sep_sets.clear()
        self._adj_directed.clear()
        self._adj_undirected.clear()
        self.test_results.clear()

        self._test_target_edges(df, drivers)
        self._test_driver_skeleton(df, drivers)

        self.nodes_ = [*list(drivers), self.target]
        return self

    def get_directed_edges(self) -> list[tuple[str, str]]:
        """Return a list of directed edges ``(u, v)``.

        .. code-block:: python

            edges = model.get_directed_edges()
        """
        return sorted(self._adj_directed)

    def get_undirected_edges(self) -> list[tuple[str, str]]:
        """Return a list of undirected edges ``(u, v)``.

        .. code-block:: python

            edges = model.get_undirected_edges()
        """
        return sorted(self._adj_undirected)

    def get_test_results(self, x: str, y: str) -> list[dict[str, float]]:
        """Return all ΔBIC test result dicts for the unordered pair ``(x, y)``.

        .. code-block:: python

            tests = model.get_test_results("A", "Y")
        """
        return [v for (xi, yi, _), v in self.test_results.items() if {xi, yi} == {x, y}]

    def summary(self) -> str:
        """Return a human-readable summary of edges and number of CI tests.

        .. code-block:: python

            print(model.summary())
        """
        lines = ["=== Directed edges ==="]
        for u, v in self.get_directed_edges():
            lines.append(f"{u} -> {v}")
        lines.append("=== Undirected edges ===")
        for u, v in self.get_undirected_edges():
            lines.append(f"{u} -- {v}")
        lines.append("=== Number of CI tests run ===")
        lines.append(str(len(self.test_results)))
        return "\n".join(lines)

    def to_digraph(self) -> str:
        """Export the learned graph as a DOT-format string for Graphviz.

        .. code-block:: python

            dot = model.to_digraph()
        """
        lines = ["digraph G {", "  node [shape=ellipse];"]
        for n in self.nodes_:
            if n == self.target:
                lines.append(f'  "{n}" [style=filled, fillcolor="#eef5ff"];')
            else:
                lines.append(f'  "{n}";')
        for u, v in self.get_directed_edges():
            lines.append(f'  "{u}" -> "{v}";')
        for u, v in self.get_undirected_edges():
            lines.append(f'  "{u}" -> "{v}" [style=dashed, dir=none];')
        lines.append("}")
        return "\n".join(lines)


class TBF_FCI:
    r"""
    Target-first Bayes Factor Temporal PC.

    This is a time-series–adapted version of TBF-PC. It combines ideas from
    temporal FCI/PCMCI with a Bayes-factor ΔBIC conditional independence test.

    For each test :math:`X \perp Y \mid S`, compare:

    .. math::

        M_0 : Y \sim S
        \\
        M_1 : Y \sim S + X

    with BIC scores

    .. math::

        \mathrm{BIC}(M) = n \log\!\left(\tfrac{\mathrm{RSS}}{n}\right)
                          + k \log(n),

    and Bayes factor approximation

    .. math::

        \log \mathrm{BF}_{10} \approx -\tfrac{1}{2}
        \left[ \mathrm{BIC}(M_1) - \mathrm{BIC}(M_0) \right].

    Declare independence if :math:`\mathrm{BF}_{10} < \tau`.

    Parameters
    ----------
    target : str
        Name of the target variable (at time t).
    target_edge_rule : {"any", "conservative", "fullS"}
        Rule for keeping lagged → target edges.
    bf_thresh : float, default=1.0
        Declare independence if BF10 < bf_thresh.
    forbidden_edges : list of tuple[str, str], optional
        Prior knowledge: edges to exclude.
    max_lag : int, default=2
        Maximum lag to include (t-1, t-2, …).
    allow_contemporaneous : bool, default=True
        Whether to allow contemporaneous edges at time t.

    Examples
    --------
    **1. Basic usage**

    .. code-block:: python

        rng = np.random.default_rng(7)
        n = 1000
        e = lambda: rng.normal(size=n)

        A = e()
        B = 0.8*np.roll(A,1) + e()
        Y = 0.6*np.roll(B,1) + e()

        df = pd.DataFrame({"A":A,"B":B,"Y":Y})

        model = TBF_FCI(target="Y", max_lag=2, target_edge_rule="fullS")
        model.fit(df, drivers=["A","B"])

        print(model.summary())
        print(model.to_digraph(collapsed=True))

    **2. Forbidding edges**

    .. code-block:: python

        model = TBF_FCI(
            target="Y",
            max_lag=2,
            target_edge_rule="any",
            forbidden_edges=[("A[t-1]","Y[t]")]
        )
        model.fit(df, drivers=["A","B"])
        print(model.to_digraph(collapsed=False))

    **3. Conservative edge rule**

    .. code-block:: python

        model = TBF_FCI(target="Y", max_lag=2, target_edge_rule="conservative")
        model.fit(df, drivers=["A","B"])
        print(model.collapsed_summary())

    References
    ----------
    - Spirtes, Glymour, Scheines (2000). *Causation, Prediction, and Search*. MIT Press. [PC/FCI foundations]
    - Spirtes (2001). "An Anytime Algorithm for Causal Inference." [FCI algorithm]
    - Colombo & Maathuis (2011). "Learning high-dimensional DAGs with latent and selection variables." [RFCI]
    - Entner & Hoyer (2010). "On causal discovery from time series data using FCI." [tsFCI]
    - Chang Gong, Di Yao, Chuzhe Zhang, Wenbin Li, Jingping Bi. (2023). *Causal Discovery from Temporal Data: A Survey*.
    - Kass & Raftery (1995). "Bayes Factors." JASA. [ΔBIC ≈ 2 log BF]
    """

    def __init__(
        self,
        target: str,
        *,
        target_edge_rule: str = "any",
        bf_thresh: float = 1.0,
        forbidden_edges: Sequence[tuple[str, str]] | None = None,
        max_lag: int = 2,
        allow_contemporaneous: bool = True,
    ):
        warnings.warn(
            "TBF_FCI is experimental and its API may change; use with caution.",
            UserWarning,
            stacklevel=2,
        )
        if not isinstance(target, str) or not target:
            raise ValueError("target must be a non-empty string")
        allowed_rules = {"any", "conservative", "fullS"}
        if target_edge_rule not in allowed_rules:
            raise ValueError(f"target_edge_rule must be one of {allowed_rules}")
        if not isinstance(bf_thresh, (int, float)) or bf_thresh <= 0:
            raise ValueError("bf_thresh must be a positive float")
        if not isinstance(max_lag, int) or max_lag < 0:
            raise ValueError("max_lag must be a non-negative integer")

        self.target = target
        self.target_edge_rule = target_edge_rule
        self.bf_thresh = float(bf_thresh)
        self.max_lag = int(max_lag)
        self.allow_contemporaneous = allow_contemporaneous
        self.forbidden_edges: set[tuple[str, str]] = self._expand_edges(forbidden_edges)

        # Internal state
        self.sep_sets: dict[tuple[str, str], set[str]] = {}
        self._adj_directed: set[tuple[str, str]] = set()
        self._adj_undirected: set[tuple[str, str]] = set()
        self.nodes_: list[str] = []
        self.test_results: dict[tuple[str, str, frozenset], dict[str, float]] = {}

        # Shared response vector for symbolic BIC
        self.y_sh = pytensor.shared(np.zeros(1, dtype="float64"), name="y_sh")
        self._bic_fn = self._build_symbolic_bic_fn()

    # ---------------------------------------------------------------------
    # Utilities for lagged names
    # ---------------------------------------------------------------------
    def _lag_name(self, var: str, lag: int) -> str:
        """Return canonical lagged variable name like ``X[t-2]`` or ``X[t]``."""
        return f"{var}[t-{lag}]" if lag > 0 else f"{var}[t]"

    def _parse_lag(self, name: str) -> tuple[str, int]:
        """Parse a lagged name into base and lag integer, defaulting to 0."""
        if "[t-" in name:
            base, lagpart = name.split("[t-")
            return base, int(lagpart[:-1])
        elif "[t]" in name:
            return name.replace("[t]", ""), 0
        else:
            return name, 0

    def _expand_edges(
        self, forbidden_edges: Sequence[tuple[str, str]] | None
    ) -> set[tuple[str, str]]:
        """Expand collapsed forbidden edges (X, Y) into all lagged forms."""
        expanded = set()
        if forbidden_edges:
            for u, v in forbidden_edges:
                if "[t" in u or "[t" in v:
                    # Already explicit
                    expanded.add((u, v))
                else:
                    # Expand across all lags
                    for lag_u in range(0, self.max_lag + 1):
                        for lag_v in range(0, self.max_lag + 1):
                            u_name = f"{u}[t-{lag_u}]" if lag_u > 0 else f"{u}[t]"
                            v_name = f"{v}[t-{lag_v}]" if lag_v > 0 else f"{v}[t]"
                            expanded.add((u_name, v_name))
        return expanded

    def _build_lagged_df(
        self, df: pd.DataFrame, variables: Sequence[str]
    ) -> pd.DataFrame:
        """Construct a time-unrolled DataFrame for variables and lags up to ``max_lag``."""
        frames = {}
        for lag in range(0, self.max_lag + 1):
            shifted = df[variables].shift(lag)
            shifted.columns = [self._lag_name(c, lag) for c in shifted.columns]
            frames[lag] = shifted
        out = pd.concat(frames.values(), axis=1).iloc[self.max_lag :]  # drop NaNs
        return out.astype("float64")

    def _admissible_cond_set(
        self, all_vars: Sequence[str], x: str, y: str
    ) -> list[str]:
        """Return admissible conditioning variables for a CI test of ``x`` and ``y``."""
        _, lag_x = self._parse_lag(x)
        _, lag_y = self._parse_lag(y)
        max_time = min(lag_x, lag_y)
        keep = []
        for z in all_vars:
            if z in (x, y):
                continue
            _, lag_z = self._parse_lag(z)
            if lag_z >= max_time:  # same time or earlier
                keep.append(z)
        return keep

    # ---------------------------------------------------------------------
    # Graph utils
    # ---------------------------------------------------------------------
    def _key(self, u: str, v: str) -> tuple[str, str]:
        """Return a sorted 2-tuple key for an undirected edge between ``u`` and ``v``."""
        return (u, v) if u <= v else (v, u)

    def _set_sep(self, u: str, v: str, S: Sequence[str]) -> None:
        """Record the separation set ``S`` for the node pair ``(u, v)``."""
        self.sep_sets[self._key(u, v)] = set(S)

    def _has_forbidden(self, u: str, v: str) -> bool:
        """Return True if edge ``u—v`` is forbidden in either direction."""
        return (u, v) in self.forbidden_edges or (v, u) in self.forbidden_edges

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
        ):
            self._adj_undirected.add(self._key(u, v))

    def _remove_all(self, u: str, v: str) -> None:
        """Remove any edge (directed or undirected) between ``u`` and ``v``."""
        self._adj_undirected.discard(self._key(u, v))
        self._adj_directed.discard((u, v))
        self._adj_directed.discard((v, u))

    # ---------------------------------------------------------------------
    # Statistical methods
    # ---------------------------------------------------------------------
    def _build_symbolic_bic_fn(self):
        """Build and compile a function to compute BIC for a design matrix and sample size."""
        X = tt.matrix("X")
        n = tt.iscalar("n")
        beta = tt.nlinalg.pinv(X) @ self.y_sh
        resid = self.y_sh - X @ beta
        rss = tt.sum(resid**2)
        k = X.shape[1]
        bic = n * tt.log(rss / n) + k * tt.log(n)
        return pytensor.function([X, n], bic)

    def _ci_independent(
        self, df: pd.DataFrame, x: str, y: str, cond: Sequence[str]
    ) -> bool:
        """Return True if ΔBIC indicates independence of ``x`` and ``y`` given ``cond``."""
        if self._has_forbidden(x, y):
            return True
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
        result = {
            "bic0": bic0,
            "bic1": bic1,
            "delta_bic": delta_bic,
            "logBF10": logBF10,
            "BF10": BF10,
            "independent": BF10 < self.bf_thresh,
            "conditioning_set": list(cond),
        }
        self.test_results[(x, y, frozenset(cond))] = result
        return result["independent"]

    # ---------------------------------------------------------------------
    # Algorithm phases
    # ---------------------------------------------------------------------
    def _stageA_target_lagged(self, L: pd.DataFrame, drivers: Sequence[str]) -> None:
        """Phase 1: test lagged driver → target edges per ``target_edge_rule``."""
        y = self._lag_name(self.target, 0)
        all_cols = list(L.columns)
        for v in drivers:
            for lag in range(1, self.max_lag + 1):
                x = self._lag_name(v, lag)
                cand = self._admissible_cond_set(all_cols, x, y)
                max_k = min(3, len(cand))
                all_sets = [
                    S for k in range(max_k + 1) for S in it.combinations(cand, k)
                ]
                if self.target_edge_rule == "fullS":
                    all_sets = [tuple(cand)]
                if self.target_edge_rule == "any":
                    keep = True
                    for S in all_sets:
                        if self._ci_independent(L, x, y, S):
                            self._set_sep(x, y, S)
                            keep = False
                            break
                    if keep:
                        self._add_directed(x, y)
                    else:
                        self._remove_all(x, y)
                elif self.target_edge_rule == "conservative":
                    indep_all = True
                    for S in all_sets:
                        if not self._ci_independent(L, x, y, S):
                            indep_all = False
                        else:
                            self._set_sep(x, y, S)
                    if indep_all:
                        self._remove_all(x, y)
                    else:
                        self._add_directed(x, y)
                elif self.target_edge_rule == "fullS":
                    S = all_sets[0]
                    if self._ci_independent(L, x, y, S):
                        self._set_sep(x, y, S)
                        self._remove_all(x, y)
                    else:
                        self._add_directed(x, y)

    def _stageA_driver_lagged(self, L: pd.DataFrame, drivers: Sequence[str]) -> None:
        """Phase 1b: build driver lagged skeleton via pairwise CI tests."""
        cols = [c for c in L.columns if not c.startswith(self.target)]
        for xi, xj in it.combinations(cols, 2):
            _, li = self._parse_lag(xi)
            _, lj = self._parse_lag(xj)
            if li == 0 and lj == 0:  # contemporaneous handled later
                continue
            cand = self._admissible_cond_set(
                [*cols, self._lag_name(self.target, 0)], xi, xj
            )
            max_k = min(3, len(cand))
            dependent, found_sep = True, False
            for k in range(max_k + 1):
                for S in it.combinations(cand, k):
                    if self._ci_independent(L, xi, xj, S):
                        self._set_sep(xi, xj, S)
                        dependent = False
                        found_sep = True
                        break
                if found_sep:
                    break
            if dependent:
                self._add_undirected(xi, xj)
            else:
                self._remove_all(xi, xj)

    def _parents_of(self, node: str) -> list[str]:
        """Return the list of parents of ``node`` from directed edges."""
        return [u for (u, v) in self._adj_directed if v == node]

    def _stageB_contemporaneous(self, L: pd.DataFrame, drivers: Sequence[str]) -> None:
        """Phase 2: test contemporaneous (t) relations among variables at time t."""
        y_nodes = [self._lag_name(v, 0) for v in [*drivers, self.target]]
        for xi, xj in it.combinations(y_nodes, 2):
            base_S = list(set(self._parents_of(xi) + self._parents_of(xj)))
            cand_extra = [z for z in y_nodes if z not in (xi, xj)]
            max_k = 2
            dependent, found_sep = True, False
            for k in range(max_k + 1):
                for extra in it.combinations(cand_extra, k):
                    S = tuple(sorted(set(base_S).union(extra)))
                    if self._ci_independent(L, xi, xj, S):
                        self._set_sep(xi, xj, S)
                        dependent = False
                        found_sep = True
                        break
                if found_sep:
                    break
            if dependent:
                self._add_undirected(xi, xj)
            else:
                self._remove_all(xi, xj)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, drivers: Sequence[str]):
        """Run temporal causal discovery on ``df`` with lag expansion and contemporaneous phase.

        This learns lagged directed edges into the target, lagged driver skeleton,
        and optionally contemporaneous relations at time t.

        .. code-block:: python

            model = TBF_FCI(target="Y", max_lag=1)
            model.fit(df, drivers=["X1", "X2"])  # returns self
        """
        self.sep_sets.clear()
        self._adj_directed.clear()
        self._adj_undirected.clear()
        self.test_results.clear()
        all_vars = [*list(drivers), self.target]
        L = self._build_lagged_df(df, all_vars)
        self.nodes_ = list(L.columns)
        self._stageA_target_lagged(L, drivers)
        self._stageA_driver_lagged(L, drivers)
        if self.allow_contemporaneous:
            self._stageB_contemporaneous(L, drivers)
        return self

    def collapsed_summary(self):
        """Summarize the time-unrolled graph into a driver-level DAG.

        Collapse lagged graph into a driver DAG summary. Returns edges of the
        form (X, Y, lag) for directed, and (X, Y) for contemporaneous undirected.

        Returns
        -------
        collapsed_directed : list[tuple[str, str, int]]
            List of directed edges in the form (X, Y, lag).
        collapsed_undirected : list[tuple[str, str]]
            List of undirected edges.

        .. code-block:: python

            collapsed_directed, collapsed_undirected = model.collapsed_summary()
            print(collapsed_directed)
            print(collapsed_undirected)
        """
        collapsed_directed = []
        for u, v in self._adj_directed:
            base_u, lag_u = self._parse_lag(u)
            base_v, lag_v = self._parse_lag(v)
            if lag_v == 0:  # effects into present
                collapsed_directed.append((base_u, base_v, lag_u))

        collapsed_undirected = []
        for u, v in self._adj_undirected:
            base_u, lag_u = self._parse_lag(u)
            base_v, lag_v = self._parse_lag(v)
            if lag_u == lag_v == 0:  # contemporaneous at t
                collapsed_undirected.append((base_u, base_v))

        return collapsed_directed, collapsed_undirected

    def get_directed_edges(self) -> list[tuple[str, str]]:
        """Return a list of directed edges ``(u, v)``.

        .. code-block:: python

            edges = model.get_directed_edges()
        """
        return sorted(self._adj_directed)

    def get_undirected_edges(self) -> list[tuple[str, str]]:
        """Return a list of undirected edges ``(u, v)``.

        .. code-block:: python

            edges = model.get_undirected_edges()
        """
        return sorted(self._adj_undirected)

    def summary(self) -> str:
        """Return a human-readable summary of edges and number of CI tests.

        .. code-block:: python

            print(model.summary())
        """
        lines = ["=== Directed edges ==="]
        for u, v in self.get_directed_edges():
            lines.append(f"{u} -> {v}")
        lines.append("=== Undirected edges ===")
        for u, v in self.get_undirected_edges():
            lines.append(f"{u} -- {v}")
        lines.append("=== Number of CI tests run ===")
        lines.append(str(len(self.test_results)))
        return "\n".join(lines)

    def to_digraph(self, collapsed: bool = True) -> str:
        """
        Export graph as DOT format string for Graphviz.

        Parameters
        ----------
        collapsed : bool, default=True
            If False: show full time-unrolled graph with explicit lag variables.
            If True: show collapsed DAG among base variables with lag annotations.

        Returns
        -------
        str
            DOT format string suitable for Graphviz rendering.

        Examples
        --------
        .. code-block:: python

            # Full time-unrolled graph
            dot_str = model.to_digraph(collapsed=False)
            print(dot_str)

            # Collapsed summary with lag annotations
            dot_str = model.to_digraph(collapsed=True)
            print(dot_str)
        """
        lines = ["digraph G {", "  node [shape=ellipse];"]

        if not collapsed:
            # --- original time-unrolled graph ---
            for n in self.nodes_:
                if n == self._lag_name(self.target, 0):
                    lines.append(f'  "{n}" [style=filled, fillcolor="#eef5ff"];')
                else:
                    lines.append(f'  "{n}";')
            for u, v in self.get_directed_edges():
                lines.append(f'  "{u}" -> "{v}";')
            for u, v in self.get_undirected_edges():
                lines.append(f'  "{u}" -> "{v}" [style=dashed, dir=none];')

        else:
            # --- collapsed summary graph ---
            directed, undirected = self.collapsed_summary()

            # nodes: only base variable names
            base_nodes = {self._parse_lag(n)[0] for n in self.nodes_}
            for n in base_nodes:
                if n == self.target:
                    lines.append(f'  "{n}" [style=filled, fillcolor="#eef5ff"];')
                else:
                    lines.append(f'  "{n}";')

            # directed edges with lag label
            for u, v, lag in directed:
                lines.append(f'  "{u}" -> "{v}" [label="lag {lag}"];')

            # contemporaneous undirected edges
            for u, v in undirected:
                lines.append(f'  "{u}" -> "{v}" [style=dashed, dir=none, label="t"];')

        lines.append("}")
        return "\n".join(lines)


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
