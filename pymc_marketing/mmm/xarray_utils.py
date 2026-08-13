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
"""Neutral xarray dimension and aggregation helpers for summary and plotting."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr

__all__ = [
    "_apply_aggregation",
    "_dims_to_sel_kwargs",
    "_ensure_chain_draw_dims",
    "_select_dims",
    "_validate_dims",
]


def _dims_to_sel_kwargs(
    dims: dict[str, Any] | None,
) -> dict[str, Any]:
    """Convert scalar dim values to single-element lists for ``.sel()``."""
    if not dims:
        return {}
    return {
        k: v if isinstance(v, (list, tuple, np.ndarray)) else [v]
        for k, v in dims.items()
    }


def _validate_dims(
    dataset: xr.Dataset | xr.DataArray,
    dims: dict[str, Any] | None,
) -> None:
    """Validate that ``dims`` keys and values exist in ``dataset`` coordinates."""
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


def _select_dims[XarrayT: (xr.Dataset, xr.DataArray)](
    data: XarrayT,
    dims: dict[str, Any] | None,
    allow_missing: bool = False,
) -> XarrayT:
    """Validate dimension filters and apply ``.sel()`` in one step."""
    if not dims:
        return data

    if allow_missing:
        filtered_dims = {k: v for k, v in dims.items() if k in data.dims}
        if not filtered_dims:
            return data
    else:
        filtered_dims = dims

    _validate_dims(data, filtered_dims)
    sel_kwargs = _dims_to_sel_kwargs(filtered_dims)
    return data.sel(**sel_kwargs)


def _ensure_chain_draw_dims(curves: xr.DataArray) -> xr.DataArray:
    """Ensure curves have ``(chain, draw)`` dimensions for ArviZ compatibility."""
    if "chain" in curves.dims and "draw" in curves.dims:
        return curves.copy()

    if "sample" not in curves.dims:
        raise ValueError(
            "Curves must have either ('chain', 'draw') or 'sample' dimensions. "
            f"Got: {list(curves.dims)}"
        )

    if "chain" in curves.coords and "draw" in curves.coords:
        return curves.unstack("sample")

    n_samples = curves.sizes["sample"]
    return (
        curves.assign_coords(chain=("sample", np.zeros(n_samples, dtype=int)))
        .assign_coords(draw=("sample", np.arange(n_samples)))
        .set_index(sample=["chain", "draw"])
        .unstack("sample")
    )


def _apply_aggregation(
    da: xr.DataArray,
    aggregation: dict[str, str | list[str]] | None,
) -> xr.DataArray:
    """Apply a single aggregation operation to *da*."""
    if not aggregation:
        return da

    if len(aggregation) > 1:
        raise ValueError(
            f"Only a single aggregation operation is supported, "
            f"got {len(aggregation)}: {list(aggregation)}."
        )

    op, dim_spec = next(iter(aggregation.items()))
    dims_list = [dim_spec] if isinstance(dim_spec, str) else list(dim_spec)

    if op == "sum":
        return da.sum(dim=dims_list)
    if op == "mean":
        return da.mean(dim=dims_list)
    raise ValueError(f"Unknown aggregation operation '{op}'. Supported: 'sum', 'mean'.")
