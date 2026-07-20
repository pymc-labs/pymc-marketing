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

The causal-discovery tooling — :class:`TBFPC` (the Target-first Bayes Factor
PC algorithm), :class:`BuildModelFromDAG` (build a model straight from a DAG),
and the :class:`TestResult` record — has moved to the standalone
`pathmc <https://github.com/pymc-labs/pathmc>`_ library, which now owns
PyMC-Labs' path-analysis / structural-causal-modeling stack. They are
re-exported here for backwards compatibility and imported **lazily**: importing
this module does not require ``pathmc``, but accessing those names does (install
it via ``pip install pymc-marketing[dag]``).

:class:`CausalGraphModel` (dowhy-based backdoor adjustment) stays here because
it is used directly by the core MMM, and pathmc provides its own native
identification rather than wrapping dowhy.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pandas as pd

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


if TYPE_CHECKING:
    from pathmc import (
        TBFPC,
        BuildModelFromDAG,
        TestResult,
    )


__all__ = ["TBFPC", "BuildModelFromDAG", "CausalGraphModel", "TestResult"]

# Names that now live in pathmc and are re-exported lazily from this module.
_PATHMC_REEXPORTS = frozenset({"TBFPC", "BuildModelFromDAG", "TestResult"})


def __getattr__(name: str):
    """Lazily resolve causal-discovery names from pathmc (PEP 562).

    Keeps this module importable without ``pathmc`` (the core MMM only needs
    :class:`CausalGraphModel`); the discovery tools are fetched from pathmc on
    first access, with an actionable error if it is not installed.
    """
    if name in _PATHMC_REEXPORTS:
        try:
            import pathmc
        except ImportError as exc:  # pragma: no cover - exercised via tests
            raise ImportError(
                f"'{name}' has moved to the 'pathmc' library. Install it with "
                "'pip install pymc-marketing[dag]'. Note that pathmc requires "
                "PyMC >= 6."
            ) from exc
        return getattr(pathmc, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
        .. [1] Causal Inference in Statistics: A Primer.
           By Judea Pearl, Madelyn Glymour, Nicholas P. Jewell · 2016.
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
