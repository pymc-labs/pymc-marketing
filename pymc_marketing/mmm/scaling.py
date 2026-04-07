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
"""Scaling configuration for the MMM."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date, datetime
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import ConfigDict, Field, field_validator, model_validator

from pymc_marketing.serialization import SerializableBaseModel, serialization

_FIXED_SCALING_XARRAY_KIND = "xarray.DataArray"


def panel_channel_fixed_scaling_remaining_dims(
    panel_dims: tuple[str, ...],
    scaling_dims: tuple[str, ...],
) -> tuple[str, ...]:
    """Non-date dims of channel data left after reduction over ``date`` and *scaling_dims*."""
    reduced = frozenset(scaling_dims)
    return tuple(d for d in (*panel_dims, "channel") if d not in reduced)


def _jsonable_coord_values(values: Any) -> list[Any]:
    """Convert a coordinate vector to JSON-serializable Python lists."""
    arr = np.asarray(values)
    out: list[Any] = []
    for v in arr.tolist():
        if isinstance(v, (pd.Timestamp, datetime, date, np.datetime64)):
            out.append(str(pd.Timestamp(v).isoformat()))
        else:
            out.append(v)
    return out


def _serialize_fixed_scaling_value(
    value: float | dict[str, float] | xr.DataArray,
) -> Any:
    if isinstance(value, xr.DataArray):
        coords_payload: dict[str, list[Any]] = {}
        for dim in value.dims:
            coords_payload[str(dim)] = _jsonable_coord_values(value.coords[dim].values)
        return {
            "__fixed_scaling_kind__": _FIXED_SCALING_XARRAY_KIND,
            "dims": [str(d) for d in value.dims],
            "coords": coords_payload,
            "data": np.asarray(value.values).tolist(),
            "name": value.name,
        }
    return value


def _dataarray_from_fixed_scaling_payload(payload: dict[str, Any]) -> xr.DataArray:
    """Reconstruct a DataArray from :func:`_serialize_fixed_scaling_value` output."""
    return xr.DataArray(
        data=np.asarray(payload["data"], dtype=float),
        dims=tuple(payload["dims"]),
        coords=dict(payload["coords"]),
        name=payload.get("name"),
    )


def _maybe_deserialize_fixed_scaling_value(
    value: Any,
) -> float | dict[str, float] | xr.DataArray:
    if (
        isinstance(value, dict)
        and value.get("__fixed_scaling_kind__") == _FIXED_SCALING_XARRAY_KIND
    ):
        return _dataarray_from_fixed_scaling_payload(value)
    return value


class VariableScaling(SerializableBaseModel, ABC):
    """Abstract base for scaling a variable.

    The scaling through the dimension of ``'date'`` is assumed and doesn't need
    to be specified.

    Concrete subclasses:

    - :class:`DataDerivedScaling` -- scale by a statistic of the data
      (``"max"`` or ``"mean"``), computed at fit time.
    - :class:`FixedScaling` -- use a user-supplied constant that stays the
      same across model refreshes.

    Parameters
    ----------
    dims : str or tuple of str
        The dimensions to perform the operation through (``"date"`` is always
        included implicitly).
    """

    dims: str | tuple[str, ...] = Field(
        ...,
        description="The dimensions to perform operation through.",
    )

    @abstractmethod
    def scaling_description(self) -> str:
        """Human-readable summary of the scaling strategy (e.g. for logging)."""
        ...

    @model_validator(mode="after")
    def _validate_dims(self) -> Self:
        if isinstance(self.dims, str):
            self.dims = (self.dims,)

        if "date" in self.dims:
            raise ValueError("dim of 'date' of is already assumed in the model.")

        if len(set(self.dims)) != len(self.dims):
            raise ValueError("dims must be unique.")

        return self


class DataDerivedScaling(VariableScaling):
    """Scale by a statistic of the data, computed at fit time.

    Parameters
    ----------
    method : ``"max"`` | ``"mean"``
        The scaling method.
    dims : str or tuple of str
        The dimensions to perform the operation through (``"date"`` is always
        included implicitly).

    Examples
    --------
    Max-absolute scaling (default behaviour):

    .. code-block:: python

        DataDerivedScaling(method="max", dims=())

    Mean-absolute scaling across a custom dimension:

    .. code-block:: python

        DataDerivedScaling(method="mean", dims=("country",))
    """

    method: Literal["max", "mean"] = Field(..., description="The scaling method.")

    def scaling_description(self) -> str:
        """Human-readable summary of the scaling strategy."""
        return f"data-derived ({self.method})"


class FixedScaling(VariableScaling):
    """Use a user-supplied constant that stays the same across model refreshes.

    Parameters
    ----------
    dims : str or tuple of str
        The dimensions to perform the operation through (``"date"`` is always
        included implicitly).
    value : float or dict[str, float] or xarray.DataArray
        Fixed scaling constant(s). A single ``float`` applies uniformly.

        A ``dict`` maps **coordinate labels along the single remaining
        dimension** after reducing over ``date`` and ``dims`` (see the
        multidimensional MMM). If more than one non-reduced dimension remains,
        use an :class:`xarray.DataArray` whose dimensions broadcast to that
        grid (e.g. a vector over ``country`` when the media grid is
        ``country`` × ``channel``). All values must be positive; NaNs are not
        allowed.

    Examples
    --------
    Fixed scalar scaling for production stability:

    .. code-block:: python

        FixedScaling(dims=(), value=10_000.0)

    Per-dimension fixed scaling (multidimensional MMM):

    .. code-block:: python

        FixedScaling(
            dims=("country",),
            value={"US": 50_000, "UK": 30_000},
        )

    Multi-dimensional fixed scale (e.g. country × channel) with xarray:

    .. code-block:: python

        import xarray as xr

        FixedScaling(
            dims=(),
            value=xr.DataArray(
                [[1e3, 2e3], [3e3, 4e3]],
                dims=("country", "channel"),
                coords={"country": ["US", "UK"], "channel": ["tv", "search"]},
            ),
        )

    Long-format table via :meth:`from_long_dataframe`:

    .. code-block:: python

        FixedScaling.from_long_dataframe(
            dims=(),
            df=long_df,
            value_col="scale",
            dim_cols=["country", "channel"],
        )
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    value: float | dict[str, float] | xr.DataArray = Field(
        ...,
        description="Fixed scaling constant(s). All values must be positive.",
    )

    @property
    def method(self) -> str:
        """Return the scaling method name."""
        return "fixed"

    def scaling_description(self) -> str:
        """Human-readable summary of the scaling strategy."""
        return "fixed constant"

    @field_validator("value", mode="before")
    @classmethod
    def _reject_bool(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError(
                "FixedScaling.value does not accept bool; use a numeric scalar."
            )
        return v

    @model_validator(mode="after")
    def _validate_value(self) -> Self:
        if isinstance(self.value, dict):
            for key, val in self.value.items():
                if math.isnan(val) or val <= 0:
                    raise ValueError(
                        f"All fixed scaling values must be positive and non-NaN, "
                        f"got {val} for key '{key}'."
                    )
        elif isinstance(self.value, xr.DataArray):
            arr = np.asarray(self.value.values, dtype=float)
            if np.isnan(arr).any():
                raise ValueError("Fixed scaling DataArray must not contain NaN values.")
            if (arr <= 0).any():
                raise ValueError(
                    "All values in a fixed scaling DataArray must be positive."
                )
        elif isinstance(self.value, (int, float, np.floating, np.integer)):
            if float(self.value) <= 0:
                raise ValueError(
                    f"Fixed scaling value must be positive, got {self.value}."
                )
        else:
            raise TypeError(
                "FixedScaling.value must be a positive float, dict[str, float], "
                f"or xarray.DataArray, got {type(self.value).__name__}."
            )
        return self

    @classmethod
    def from_long_dataframe(
        cls,
        dims: str | tuple[str, ...],
        df: pd.DataFrame,
        *,
        value_col: str,
        dim_cols: Sequence[str],
    ) -> Self:
        """Build fixed scaling from a long table (one row per coordinate combination).

        Parameters
        ----------
        dims
            Passed through to :class:`FixedScaling`.
        df
            Data frame with columns ``dim_cols`` and ``value_col``.
        value_col
            Column name for the positive scale values.
        dim_cols
            Column names that identify the grid (order defines ``DataArray`` dims).
        """
        s = df.set_index(list(dim_cols))[value_col]
        if s.index.duplicated().any():
            raise ValueError(
                f"Duplicate coordinate rows found in columns {list(dim_cols)}. "
                "Each coordinate combination must appear exactly once."
            )
        da = s.to_xarray()
        ordered = da.transpose(*dim_cols)
        return cls(dims=dims, value=ordered)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for :mod:`pymc_marketing.serialization` (handles DataArray)."""
        return {
            "dims": list(self.dims),
            "value": _serialize_fixed_scaling_value(self.value),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Deserialize; restores encoded :class:`xarray.DataArray` values."""
        filtered = {k: v for k, v in data.items() if k != "__type__"}
        if "value" in filtered:
            filtered["value"] = _maybe_deserialize_fixed_scaling_value(
                filtered["value"]
            )
        if "dims" in filtered and isinstance(filtered["dims"], list):
            filtered["dims"] = tuple(filtered["dims"])
        return cls.model_validate(filtered)


def _validate_fixed_scaling_keys(
    scaling: VariableScaling,
    valid_labels: list[str],
    variable_name: str,
) -> None:
    """Check that dict-valued FixedScaling keys match the expected labels.

    Parameters
    ----------
    scaling : VariableScaling
        The scaling instance to validate.
    valid_labels : list[str]
        The expected coordinate labels (e.g. channel column names).
    variable_name : str
        Human-readable name for error messages (e.g. ``"channel"``).

    Raises
    ------
    ValueError
        If the scaling is a dict-valued :class:`FixedScaling` whose keys
        don't match *valid_labels*.
    """
    if not isinstance(scaling, FixedScaling):
        return
    if not isinstance(scaling.value, dict):
        return

    expected = set(valid_labels)
    provided = set(scaling.value.keys())
    missing = expected - provided
    extra = provided - expected

    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing keys: {sorted(missing)}")
        if extra:
            parts.append(f"unexpected keys: {sorted(extra)}")
        raise ValueError(
            f"Fixed scaling dict keys for {variable_name} do not match "
            f"the expected labels. {'; '.join(parts)}. "
            f"Expected: {sorted(expected)}."
        )


def _deserialize_variable_scaling(d: dict[str, Any]) -> VariableScaling:
    """Deserialize a VariableScaling from a dict, handling both legacy and new formats.

    Legacy format (pre-class-split) uses a ``method`` field to discriminate.
    New format uses the ``__type__`` key injected by the serialization registry.
    """
    if "__type__" in d:
        return serialization.deserialize(d)

    method = d.get("method")
    dims = tuple(d.get("dims", ()))
    if method == "fixed":
        raw_value = d["value"]
        value = _maybe_deserialize_fixed_scaling_value(raw_value)
        return FixedScaling(dims=dims, value=value)
    return DataDerivedScaling(method=method, dims=dims)


class Scaling(SerializableBaseModel):
    """Scaling configuration for the MMM.

    Parameters
    ----------
    target : VariableScaling
        Scaling configuration for the target (response) variable.
    channel : VariableScaling
        Scaling configuration for the channel (media) variables.

    Examples
    --------
    Data-derived scaling:

    .. code-block:: python

        Scaling(
            target=DataDerivedScaling(method="max", dims=()),
            channel=DataDerivedScaling(method="max", dims=()),
        )

    Fixed scaling for stable production refreshes:

    .. code-block:: python

        Scaling(
            target=FixedScaling(dims=(), value=50_000.0),
            channel=FixedScaling(dims=(), value=10_000.0),
        )
    """

    target: VariableScaling = Field(
        ...,
        description="The scaling for the target variable.",
    )
    channel: VariableScaling = Field(
        ...,
        description="The scaling for the channel variable.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_dict_values(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key in ("target", "channel"):
                val = data.get(key)
                if isinstance(val, dict):
                    data[key] = _deserialize_variable_scaling(val)
        return data

    def to_dict(self) -> dict[str, Any]:
        """Serialize with ``__type__`` keys on nested VariableScaling subclasses."""
        return {
            "target": serialization.serialize(self.target),
            "channel": serialization.serialize(self.channel),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Reconstruct from a dict, dispatching nested VariableScaling via __type__."""
        filtered = {k: v for k, v in data.items() if k != "__type__"}
        for key in ("target", "channel"):
            if key in filtered and isinstance(filtered[key], dict):
                filtered[key] = _deserialize_variable_scaling(filtered[key])
        return cls.model_validate(filtered)
