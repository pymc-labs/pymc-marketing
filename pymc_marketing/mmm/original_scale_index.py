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

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from xarray import Index
from xarray.core.indexing import IndexSelResult
from xarray.indexes import PandasIndex


def _is_scalar_indexer(idx: Any) -> bool:
    """Check whether an integer indexer reduces its dimension to a scalar."""
    if isinstance(idx, slice | xr.Variable):
        return False
    return np.ndim(idx) == 0


def _is_scalar_label(label: Any) -> bool:
    """Check whether a label-based indexer selects a single position."""
    return not isinstance(label, slice) and np.ndim(label) == 0


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

    Selection semantics
    -------------------
    ``"x"`` labels are always interpreted in the **original (spend) domain**, and
    they can only be resolved once every scale dimension has been reduced to a
    scalar, because a single spend value maps to a different position on each
    channel's axis. Select TV's curve at a spend of 2500 with

    .. code-block:: python

        curve.sel(channel="TV").sel(x=2500)

    ``"x"`` **slices and lists** may be combined with scale labels in a single
    call — the scale dimensions are resolved first and the ``"x"`` label is
    applied to the resolved spend axis:

    .. code-block:: python

        curve.sel(channel="TV", x=slice(0, 2500))

    A **scalar** ``"x"`` label cannot be combined with scale labels in one call
    (the result would silently fall back to a scaled-domain coordinate), and no
    ``"x"`` label can be used while any scale dimension is unresolved — both
    raise :class:`NotImplementedError` with guidance. Positional ``.isel`` is
    unaffected. Scale dimension labels accept scalars, lists and slices and are
    resolved through a :class:`~xarray.indexes.PandasIndex` per dimension.

    Known limitations
    -----------------
    - :meth:`~xarray.DataArray.groupby` over a scale dimension keeps **scaled**
      ``x`` on each group: groupby reduces via a one-element list, so the index
      never reaches its fully-resolved branch. Use ``.sel`` per group for
      original-domain ``x``.
    - Selecting a **list** on a scale dimension keeps the remaining curve in the
      scaled domain (the original domain is a different axis per channel).
    - To operate on the raw scaled coordinate, drop the index first:
      ``curve.drop_indexes(["channel", "x"])``.

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

        import numpy as np
        import xarray as xr

        channels = ["TV", "Radio"]
        x = np.linspace(0, 1, 100)
        channel_scale = xr.DataArray(
            [5000.0, 1200.0],
            dims=["channel"],
            coords={"channel": channels},
        )
        # y = x (linear / "infinite returns" saturation)
        da = xr.DataArray(
            np.broadcast_to(x, (len(channels), len(x))).copy(),
            dims=["channel", "x"],
            coords={"channel": channels, "x": x},
        )
        curve = da.drop_indexes(["x", "channel"]).set_xindex(
            ["x", "channel"], OriginalScaleIndex, channel_scale=channel_scale
        )
        curve.sel(channel="TV").coords["x"]  # TV's original-domain x values [0, 5000]
        curve.sel(channel="TV", x=slice(0, 2500))  # TV's curve up to a spend of 2500

    Intended use-case: attaching original-domain x coordinates to MMM saturation
    curves automatically, e.g. from ``MMM.sample_saturation_curve``
    (`#2740 <https://github.com/pymc-labs/pymc-marketing/issues/2740>`__).
    """

    def __init__(self, x_original: xr.DataArray) -> None:
        self.x_original = x_original
        self._scale_indexes: dict[str, PandasIndex] = {
            dim: PandasIndex(pd.Index(x_original.coords[dim].values), dim)
            for dim in self._scale_dims
        }

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
        """Return the coordinate variables managed by this index.

        Parameters
        ----------
        variables : dict, optional
            Variables already created by the caller. Entries for the dimensions
            managed by this index are passed through unchanged, preserving any
            ``attrs`` or dtype; missing entries are rebuilt from the underlying
            coordinates.

        Returns
        -------
        dict[str, xr.Variable]
        """
        result: dict[str, xr.Variable] = {}
        if "x" in self.x_original.coords:
            names = ["x", *self._scale_dims]
        else:
            names = list(self._scale_dims)
        for name in names:
            if variables is not None and name in variables:
                result[name] = variables[name]
            else:
                result[name] = xr.Variable(name, self.x_original.coords[name].values)
        return result

    def sel(
        self,
        labels: dict,
        method: str | None = None,
        tolerance: float | None = None,
    ) -> IndexSelResult:
        """Handle label-based selection.

        Scale dimension labels (scalars, lists and slices) are resolved to
        positions through a :class:`~xarray.indexes.PandasIndex` per dimension.
        Once every scale dimension is reduced to a scalar, the ``"x"`` coordinate
        is replaced with original-domain values backed by a
        :class:`~xarray.indexes.PandasIndex`.

        ``"x"`` labels are in the original (spend) domain. They are accepted
        alongside scale labels as slices or lists, in which case the scale
        dimensions are resolved first and the ``"x"`` label is applied to the
        resolved axis. A scalar ``"x"`` label, or any ``"x"`` label with a scale
        dimension still unresolved, raises :class:`NotImplementedError`.

        Parameters
        ----------
        labels : dict
            Selection labels. Keys may be any subset of the managed scale
            dimension names or ``"x"``.
        method : str, optional
            Look-up method for scalar labels, e.g. ``"nearest"``.
        tolerance : float, optional
            Maximum distance between the label and the matched coordinate for
            inexact look-ups.

        Returns
        -------
        IndexSelResult
        """
        scale_labels = {k: v for k, v in labels.items() if k in self._scale_indexes}
        x_labels = {k: v for k, v in labels.items() if k == "x"}
        unknown = set(labels) - set(scale_labels) - {"x"}
        if unknown:
            raise NotImplementedError(
                f"OriginalScaleIndex does not support labels: {sorted(unknown)}"
            )

        if x_labels:
            if not scale_labels:
                raise NotImplementedError(
                    "OriginalScaleIndex cannot resolve an 'x' label while the "
                    f"scale dimensions {list(self._scale_dims)} are unresolved: "
                    "'x' labels are in the original (spend) domain, which differs "
                    "per channel. Select the scale dimensions first, e.g. "
                    ".sel(channel=...).sel(x=...) or .sel(channel=..., x=...)."
                )
            if any(_is_scalar_label(val) for val in x_labels.values()):
                # A scalar 'x' label would reduce every managed dimension in one
                # call; xarray then drops the index and the original-domain x
                # coordinate cannot be carried over (it would silently fall back
                # to the scaled value).
                raise NotImplementedError(
                    "OriginalScaleIndex cannot apply a scalar 'x' label in the "
                    "same .sel() call as scale dimensions. Chain the selection "
                    "instead: .sel(channel=...).sel(x=...) — or select an 'x' "
                    "slice or list, which keeps the 'x' dimension."
                )

        dim_indexers: dict = {}
        if scale_labels:
            for dim, val in scale_labels.items():
                res = self._scale_indexes[dim].sel(
                    {dim: val}, method=method, tolerance=tolerance
                )
                dim_indexers[dim] = res.dim_indexers[dim]

            selected = self.x_original.isel(dim_indexers)
            remaining = [d for d in self._scale_dims if d in selected.dims]

            if remaining:
                return IndexSelResult(
                    dim_indexers=dim_indexers,
                    indexes={
                        "x": OriginalScaleIndex(selected),
                        **{d: OriginalScaleIndex(selected) for d in remaining},
                    },
                )
        else:
            selected = self.x_original

        # All scale dims resolved: selected is 1-D over the original-domain axis
        spend = np.asarray(selected.values)
        spend_index = PandasIndex(pd.Index(spend), "x")

        if x_labels:
            x_res = spend_index.sel(x_labels, method=method, tolerance=tolerance)
            x_pos = x_res.dim_indexers["x"]
            dim_indexers["x"] = x_pos
            return IndexSelResult(
                dim_indexers=dim_indexers,
                variables={"x": xr.Variable("x", spend[x_pos])},
                indexes={"x": PandasIndex(pd.Index(spend[x_pos]), "x")},
            )

        return IndexSelResult(
            dim_indexers=dim_indexers,
            variables={"x": xr.Variable("x", spend)},
            indexes={"x": spend_index},
        )

    def isel(self, indexers: dict) -> OriginalScaleIndex | PandasIndex | None:
        """Handle integer-based selection.

        Parameters
        ----------
        indexers : dict
            Integer indexers for ``"x"`` and/or any scale dimensions.

        Returns
        -------
        OriginalScaleIndex | PandasIndex | None
            Returns a ``PandasIndex`` on original-domain x values once all scale
            dimensions are reduced; otherwise returns a new ``OriginalScaleIndex``.
            Returns ``None`` when every managed dimension is reduced to a scalar,
            matching :meth:`~xarray.indexes.PandasIndex.isel` semantics for
            scalar selection.
        """
        x_idx = indexers.get("x", slice(None))
        scale_indexers = {d: indexers[d] for d in self._scale_dims if d in indexers}
        all_indexers = {"x": x_idx, **scale_indexers}

        for dim, idx in all_indexers.items():
            if isinstance(idx, xr.Variable) and idx.dims != (dim,):
                # An indexer introducing new dimensions prevents preserving the
                # index, matching PandasIndex behaviour.
                return None

        selected = self.x_original.isel(all_indexers)

        remaining = [d for d in self._scale_dims if d in selected.dims]
        if not remaining:
            if "x" in selected.dims:
                return PandasIndex(pd.Index(selected.values), "x")
            return None
        if _is_scalar_indexer(x_idx):
            # 'x' reduced to a scalar position: keep managing only the scale
            # dims; the scalar x coordinate is carried by positional indexing.
            selected = selected.drop_vars("x")
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
