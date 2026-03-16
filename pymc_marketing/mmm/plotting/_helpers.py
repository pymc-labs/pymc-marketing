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
"""Shared helpers for MMMPlotSuite namespace classes."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr


def _validate_dims(
    dataset: xr.Dataset,
    dims: dict[str, Any] | None,
) -> None:
    """Validate that ``dims`` keys and values exist in ``dataset`` coordinates.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset whose coordinates are checked.
    dims : dict or None
        Mapping of dimension name → value(s) to validate.
        Values may be scalars, lists, tuples, or numpy arrays.

    Raises
    ------
    ValueError
        If a dimension name is not in ``dataset.dims`` or a value
        is not present in the corresponding coordinate.
    """
    if not dims:
        return

    all_dims = list(dataset.dims)
    for key, val in dims.items():
        if key not in all_dims:
            raise ValueError(
                f"Dimension '{key}' not found in dataset dimensions. "
                f"Available: {all_dims}"
            )
        valid_values = dataset.coords[key].values
        values = val if isinstance(val, (list, tuple, np.ndarray)) else [val]
        for v in values:
            if v not in valid_values:
                raise ValueError(
                    f"Value '{v}' not found in dimension '{key}'. "
                    f"Available: {list(valid_values)}"
                )
