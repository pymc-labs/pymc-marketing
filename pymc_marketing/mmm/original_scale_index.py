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
"""Custom xarray Index for saturation curve x coordinates in original scale."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from xarray import Index
from xarray.core.indexing import IndexSelResult
from xarray.indexes import PandasIndex


class OriginalScaleIndex(Index):
    """Custom xarray Index that maps scaled x coordinates to original domain on selection.

    Stores ``x_original = x_scaled * channel_scale`` as a pre-computed DataArray
    via xarray broadcasting. Selecting on any scale dimension (e.g. ``"channel"``,
    ``"geo"``) reduces the array; once all scale dimensions are resolved, the
    ``"x"`` coordinate is replaced with original-domain values backed by a plain
    :class:`~xarray.indexes.PandasIndex`. Partial selections return a new
    ``OriginalScaleIndex`` over the remaining scale dimensions.

    ``da.xindexes["x"]`` and ``da.xindexes["channel"]`` return the same
    ``OriginalScaleIndex`` instance.

    Parameters
    ----------
    x_original : xr.DataArray
        Pre-computed original-domain x values. Must have ``"x"`` as one dimension;
        all other dimensions are treated as scale dimensions (e.g. ``("x", "channel")``
        or ``("x", "geo", "channel")``). The ``"x"`` coordinate holds the scaled
        linspace values; the DataArray values are ``x_scaled * channel_scale``.

    Examples
    --------
    Attach directly to a DataArray via :func:`~xarray.DataArray.set_xindex`:

    .. code-block:: python

        channel_scale = xr.DataArray(
            [5000.0, 1200.0],
            dims=["channel"],
            coords={"channel": ["TV", "Radio"]},
        )
        # da has dims (chain, draw, channel, x) with x in [0, 1]
        curve = da.drop_indexes(["x", "channel"]).set_xindex(
            ["x", "channel"], OriginalScaleIndex, channel_scale=channel_scale
        )
        curve.sel(channel="TV").coords["x"]  # TV's original-domain x values

    See :meth:`~pymc_marketing.mmm.MMM.sample_saturation_curve` for the primary
    use-case where this index is attached automatically.
    """

    def __init__(self, x_original: xr.DataArray) -> None:
        self.x_original = x_original

    @property
    def _scale_dims(self) -> tuple[str, ...]:
        return tuple(d for d in self.x_original.dims if d != "x")

    @classmethod
    def _from_x_and_scale(
        cls,
        x_values: np.ndarray,
        channel_scale: xr.DataArray,
    ) -> OriginalScaleIndex:
        x_da = xr.DataArray(x_values, dims=["x"], coords={"x": x_values})
        return cls(x_original=x_da * channel_scale)

    @classmethod
    def from_variables(
        cls,
        variables: dict[str, xr.Variable],
        *,
        options: dict | None = None,
    ) -> OriginalScaleIndex:
        """Create an OriginalScaleIndex from coordinate variables.

        Called by xarray when ``set_xindex([...], OriginalScaleIndex, channel_scale=da)``
        is invoked.

        Parameters
        ----------
        variables : dict[str, xr.Variable]
            Must contain an ``"x"`` variable (scaled x linspace). Scale dimension
            variables are derived from ``options["channel_scale"]``.
        options : dict, optional
            Must contain ``"channel_scale"``: an ``xr.DataArray`` with the per-channel
            scale factors and whatever dims apply.

        Returns
        -------
        OriginalScaleIndex
        """
        options = options or {}
        x_var = variables.get("x")
        channel_scale = options.get("channel_scale")
        if x_var is None:
            raise ValueError(
                f"OriginalScaleIndex requires an 'x' variable. Got: {list(variables)}"
            )
        if channel_scale is None:
            raise ValueError(
                "OriginalScaleIndex requires 'channel_scale' in options. "
                "Pass it via set_xindex(..., channel_scale=da)."
            )
        return cls._from_x_and_scale(x_var.values, channel_scale)

    def create_variables(
        self,
        variables: dict | None = None,
    ) -> dict[str, xr.Variable]:
        """Return the coordinate variables managed by this index."""
        result: dict[str, xr.Variable] = {
            "x": xr.Variable("x", self.x_original.coords["x"].values)
        }
        for dim in self._scale_dims:
            result[dim] = xr.Variable(dim, self.x_original.coords[dim].values)
        return result

    def sel(
        self,
        labels: dict,
        method: str | None = None,
        tolerance: float | None = None,
    ) -> IndexSelResult:
        """Handle label-based selection.

        Selects on scale dimensions by indexing into the pre-computed ``x_original``
        DataArray. When all scale dimensions are resolved the ``x`` coordinate is
        replaced with original-domain values. Partial selections return a new
        ``OriginalScaleIndex`` over the remaining dimensions.

        Parameters
        ----------
        labels : dict
            Selection labels. Keys may be any subset of the managed scale
            dimension names or ``"x"``.
        method : str, optional
            Passed through to PandasIndex for ``"x"`` selection.
        tolerance : float, optional
            Passed through to PandasIndex for ``"x"`` selection.

        Returns
        -------
        IndexSelResult
        """
        scale_labels = {k: v for k, v in labels.items() if k in self._scale_dims}
        x_labels = {k: v for k, v in labels.items() if k == "x"}

        if scale_labels:
            dim_indexers: dict[str, int] = {}
            for dim, val in scale_labels.items():
                arr = self.x_original.coords[dim].values
                matches = np.where(arr == val)[0]
                if not len(matches):
                    raise KeyError(f"{dim}={val!r} not found. Available: {list(arr)}")
                dim_indexers[dim] = int(matches[0])

            selected = self.x_original.sel(scale_labels)
            remaining = [d for d in self._scale_dims if d not in scale_labels]

            if not remaining:
                x_natural = selected.values  # shape (n_x,) — already in original domain
                return IndexSelResult(
                    dim_indexers=dim_indexers,
                    variables={"x": xr.Variable("x", x_natural)},
                    indexes={"x": PandasIndex(pd.Index(x_natural), "x")},
                )
            else:
                new_idx = OriginalScaleIndex(selected)
                return IndexSelResult(
                    dim_indexers=dim_indexers,
                    indexes={"x": new_idx, **{d: new_idx for d in remaining}},
                )

        if x_labels:
            x_scaled = self.x_original.coords["x"].values
            return PandasIndex(pd.Index(x_scaled), "x").sel(
                {"x": x_labels["x"]}, method=method, tolerance=tolerance
            )

        raise NotImplementedError(
            f"OriginalScaleIndex.sel does not support labels: {list(labels)}"
        )

    def isel(self, indexers: dict) -> OriginalScaleIndex | PandasIndex:
        """Handle integer-based selection.

        Parameters
        ----------
        indexers : dict
            Integer indexers for ``"x"`` and/or any scale dimensions.

        Returns
        -------
        OriginalScaleIndex | PandasIndex
            Returns a ``PandasIndex`` on original-domain x values once all scale
            dimensions are reduced; otherwise returns a new ``OriginalScaleIndex``.
        """
        x_idx = indexers.get("x", slice(None))
        scale_indexers = {d: indexers[d] for d in self._scale_dims if d in indexers}
        selected = self.x_original.isel({"x": x_idx, **scale_indexers})

        remaining = [d for d in self._scale_dims if d not in scale_indexers]
        if not remaining:
            return PandasIndex(pd.Index(selected.values), "x")
        return OriginalScaleIndex(selected)

    def equals(self, other: object) -> bool:
        """Check equality with another index."""
        return isinstance(other, OriginalScaleIndex) and self.x_original.equals(
            other.x_original
        )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"OriginalScaleIndex(scale_dims={list(self._scale_dims)}, "
            f"n_x={self.x_original.sizes['x']})"
        )
