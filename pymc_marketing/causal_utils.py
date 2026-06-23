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

The Markov-equivalence machinery has moved to the standalone
`pathmc <https://github.com/pymc-labs/pathmc>`_ library, whose
``same_markov_equivalence_class`` compares graphs via skeleton + v-structure
equality using a dependency-free DOT reader (no ``pydot`` needed). This module
keeps the historical name :func:`same_markov_equivalence_class_CPdag` as a thin
wrapper, imported lazily so this module stays importable without ``pathmc``
(install it with ``pip install pymc-marketing[dag]``).
"""

from __future__ import annotations

from typing import Protocol

__all__ = ["same_markov_equivalence_class_CPdag"]


class _SupportsSource(Protocol):
    @property
    def source(self) -> str: ...


def same_markov_equivalence_class_CPdag(
    dot1: str | _SupportsSource, dot2: str | _SupportsSource
) -> bool:
    """Determine whether two DOT graphs share a Markov equivalence class.

    Thin wrapper over :func:`pathmc.same_markov_equivalence_class`. Each
    argument may be a DOT string or an object exposing a ``.source`` attribute
    (e.g. a ``graphviz.Digraph``).

    Parameters
    ----------
    dot1, dot2 : str | object with ``.source``
        The two graphs to compare.

    Returns
    -------
    bool
        ``True`` if the graphs are Markov-equivalent, ``False`` otherwise.
    """
    try:
        from pathmc import same_markov_equivalence_class
    except ImportError as exc:  # pragma: no cover - exercised via tests
        raise ImportError(
            "same_markov_equivalence_class_CPdag now delegates to the 'pathmc' "
            "library. Install it with 'pip install pymc-marketing[dag]'. Note "
            "that pathmc requires PyMC >= 6."
        ) from exc
    return same_markov_equivalence_class(dot1, dot2)
