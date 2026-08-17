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
"""Utilities for causal reasoning and discovery.

The Markov-equivalence machinery lives in the standalone
`pathmc <https://github.com/pymc-labs/pathmc>`_ library, whose
``same_markov_equivalence_class`` compares graphs via skeleton + unshielded-collider
equality using a dependency-free DOT reader. This module keeps the historical
name :func:`same_markov_equivalence_class_CPdag` as a thin wrapper, imported
lazily so the module stays importable without ``pathmc`` (install it with
``pip install pymc-marketing[dag]``).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import networkx as nx

__all__ = ["same_markov_equivalence_class_CPdag"]

# Whitespace and comments that may precede a DOT ``graph``/``digraph`` header.
_DOT_PREAMBLE = re.compile(r"\A(?:\s|//[^\n]*|#[^\n]*|/\*.*?\*/)*", re.DOTALL)


class _SupportsSource(Protocol):
    @property
    def source(self) -> str: ...


# What pathmc's Markov-equivalence check accepts.
type Graph = str | _SupportsSource | nx.DiGraph


def _strip_dot_preamble(dot: str) -> str:
    """Drop comments and whitespace preceding a DOT ``graph``/``digraph`` header.

    ``graphviz.Digraph(comment=...)`` emits ``// <comment>`` above the header,
    while pathmc's DOT readers require the keyword to come first. Text with no
    such preamble is returned unchanged.
    """
    return dot[_DOT_PREAMBLE.match(dot).end() :]  # type: ignore[union-attr]


def _as_dot(graph: Graph) -> Graph:
    """Strip the DOT preamble off *graph*, leaving non-DOT inputs untouched."""
    if isinstance(graph, str):
        return _strip_dot_preamble(graph)
    source = getattr(graph, "source", None)
    if isinstance(source, str):
        return _strip_dot_preamble(source)
    return graph


def same_markov_equivalence_class_CPdag(dot1: Graph, dot2: Graph) -> bool:
    """Determine whether two graphs share a Markov equivalence class.

    Thin wrapper over :func:`pathmc.same_markov_equivalence_class`. Each
    argument may be a DOT string, an object exposing a ``.source`` attribute
    (e.g. a ``graphviz.Digraph``), or a :class:`networkx.DiGraph`.

    Parameters
    ----------
    dot1, dot2 : str | object with ``.source`` | networkx.DiGraph
        The two graphs to compare.

    Returns
    -------
    bool
        ``True`` if the graphs are Markov-equivalent, ``False`` otherwise.
    """
    try:
        from pathmc import same_markov_equivalence_class
    except ImportError as exc:
        raise ImportError(
            "same_markov_equivalence_class_CPdag delegates to the 'pathmc' "
            "library. Install it with 'pip install pymc-marketing[dag]'."
        ) from exc
    return same_markov_equivalence_class(_as_dot(dot1), _as_dot(dot2))
