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
"""Causal module.

Every causal algorithm reachable from here is implemented in the standalone
`pathmc <https://github.com/pymc-labs/pathmc>`_ library, which owns PyMC-Labs'
path-analysis / structural-causal-modeling stack:

* :class:`TBFPC` (Target-first Bayes Factor PC discovery), :class:`BuildModelFromDAG`
  (build a model straight from a DAG) and the :class:`TestResult` record are
  re-exported from pathmc unchanged.
* :class:`CausalGraphModel` -- what :class:`~pymc_marketing.mmm.MMM` uses to turn a
  DAG into an adjustment set -- is backed by pathmc's native backdoor
  identification (:meth:`pathmc.PathModel.adjustment_sets`).

pathmc is optional. It ships in the ``dag`` extra
(``pip install pymc-marketing[dag]``) and is imported lazily, so importing this
module -- and with it the core MMM -- never requires pathmc to be installed.
Only reaching for a causal algorithm does.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathmc import (
        TBFPC,
        BuildModelFromDAG,
        PathModel,
        TestResult,
    )


__all__ = ["TBFPC", "BuildModelFromDAG", "CausalGraphModel", "TestResult"]

# Names re-exported verbatim from pathmc.
_PATHMC_REEXPORTS = frozenset({"TBFPC", "BuildModelFromDAG", "TestResult"})


def _import_pathmc():
    """Import pathmc, raising an actionable error when it is not installed."""
    try:
        import pathmc
    except ImportError as exc:
        raise ImportError(
            "The causal tooling is backed by the 'pathmc' library. Install it "
            "with 'pip install pymc-marketing[dag]'."
        ) from exc
    return pathmc


def __getattr__(name: str):
    """Resolve the causal-discovery names from pathmc on first access (PEP 562).

    Keeps this module importable without ``pathmc``; the discovery tools are
    fetched from pathmc when they are used, with an actionable error if it is
    not installed.
    """
    if name in _PATHMC_REEXPORTS:
        return getattr(_import_pathmc(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class CausalGraphModel:
    """Represent a causal model based on a Directed Acyclic Graph (DAG).

    Provides methods to analyze causal relationships and determine the minimal adjustment set
    for backdoor adjustment between treatment and outcome variables.

    Identification is delegated to pathmc, which applies the backdoor criterion
    to the DAG: candidate sets never contain a descendant of the treatment, they
    must d-separate treatment from outcome once the treatment's incoming edges
    are removed, and only minimal sets are returned.

    Parameters
    ----------
    path_model : pathmc.PathModel
        A pathmc structural model carrying the causal DAG. Build one with
        :meth:`build_graphical_model` instead of assembling it by hand.
    treatment : list[str]
        A list of treatment variable names.
    outcome : str
        The outcome variable name.

    References
    ----------
    .. [1] https://github.com/pymc-labs/pathmc
    """

    def __init__(
        self, path_model: PathModel, treatment: list[str] | tuple[str], outcome: str
    ) -> None:
        self.path_model = path_model
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
        pathmc = _import_pathmc()
        # ``dag_to_spec`` turns each edge ``A -> B`` into a regression term; no
        # data is passed because identification reads the graph only.
        path_model = pathmc.model(pathmc.dag_to_spec(graph))
        return cls(path_model, treatment, outcome)

    def get_unique_adjustment_nodes(self) -> list[str]:
        """Compute the minimal adjustment set required for backdoor adjustment across all treatments.

        Each treatment contributes its smallest valid backdoor adjustment set --
        pathmc orders the valid sets by size and then alphabetically, so the
        choice is deterministic -- and the contributions are unioned. Treatments
        and the outcome are never part of the adjustment set.

        Returns
        -------
        list[str]
            A sorted list of unique adjustment variables needed to block all backdoor paths.

        Warns
        -----
        UserWarning
            If the effect of a treatment on the outcome cannot be identified by
            backdoor adjustment, in which case that treatment contributes
            nothing to the returned set.
        """
        adjustment_nodes: set[str] = set()
        for treatment in self.treatment:
            valid_sets = self.path_model.adjustment_sets(treatment, self.outcome)
            if not valid_sets:
                warnings.warn(
                    f"The effect of '{treatment}' on '{self.outcome}' is not "
                    "identifiable by backdoor adjustment: no set of variables "
                    "blocks every backdoor path. Estimates for this treatment "
                    "may remain confounded.",
                    stacklevel=2,
                )
                continue
            adjustment_nodes |= valid_sets[0]

        return sorted(adjustment_nodes - set(self.treatment) - {self.outcome})

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

        control_columns = sorted(common_controls - set(channel_columns))

        self.minimal_adjustment_set = control_columns + list(channel_columns)

        for column in self.adjustment_set:
            if column not in control_columns and column not in channel_columns:
                warnings.warn(
                    f"""Column {column} in adjustment set not found in data.
                    Not controlling for this may induce bias in treatment effect estimates.""",
                    stacklevel=2,
                )

        return control_columns
