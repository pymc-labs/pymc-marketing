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

"""Buffer checks shared by the caches whose entries read a cloned model's data.

A compiled entry reads the clone's copies of the data variables, which never
change again, while the source model keeps the buffers ``pm.set_data``
replaces.  Recording the source buffers a graph reads, and noticing that one
was replaced, is what lets a cache rebuild an entry instead of returning a
result compiled against stale data.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping

import numpy as np
from pymc.model.core import Model
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.basic import Variable
from pytensor.graph.traversal import ancestors


def watched_source_buffers(
    source_model: Model,
    graphs: Iterable[Variable],
    *,
    exclude: Collection[str] = (),
) -> dict[str, np.ndarray]:
    """Record the source-model data buffers ``graphs`` read, keyed by name.

    :class:`~pymc_marketing.pytensor_utils.ConditionedModel` clones the model,
    so the graphs read the clone's copies of the data variables, which never
    change again.  The source model, which ``pm.set_data`` mutates, is matched
    by name.  A graph node with no source-model counterpart, such as the shared
    posterior variables or an anonymous symbolic input, is not recorded.

    Parameters
    ----------
    source_model : Model
        The model the graphs were cloned from.
    graphs : iterable of Variable
        Graph outputs whose ancestors are scanned.
    exclude : collection of str, default ()
        Names never to record, for inputs a caller reads another way.

    Returns
    -------
    dict of str to np.ndarray
        The borrowed buffer of each source data variable the graphs read.
    """
    watched: dict[str, np.ndarray] = {}
    for node in ancestors(graphs):
        name = node.name
        if name is None or name in watched or name in exclude:
            continue
        source_var = source_model.named_vars.get(name)
        if isinstance(source_var, SharedVariable):
            watched[name] = source_var.get_value(borrow=True)
    return watched


def buffers_replaced(source_model: Model, watched: Mapping[str, np.ndarray]) -> bool:
    """Whether a source-model data buffer changed since it was recorded.

    Each watched variable costs one borrowed-buffer identity check and no value
    comparison.  ``pm.set_data`` replaces the buffer, which is what this sees;
    an in-place write to the buffer itself is not detected.  A buffer replaced
    by equal values still counts as a change, and a variable that is no longer
    a shared input counts as one too.

    Parameters
    ----------
    source_model : Model
        The model a call is about to use.
    watched : dict of str to np.ndarray
        Buffers recorded by :func:`watched_source_buffers`.

    Returns
    -------
    bool
        ``True`` when a watched source variable no longer holds the buffer
        recorded at build time, so the compiled graphs read stale values.
    """
    for name, buffer in watched.items():
        source_var = source_model.named_vars.get(name)
        if not isinstance(source_var, SharedVariable):
            return True
        if source_var.get_value(borrow=True) is not buffer:
            return True
    return False
