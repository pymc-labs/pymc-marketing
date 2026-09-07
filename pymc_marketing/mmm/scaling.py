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
"""Scaling configuration for the MMM.

Each :class:`VariableScaling` subclass follows a ``fit`` / ``transform`` /
``inverse_transform`` contract:

* :meth:`~VariableScaling.compute` -- derive scaling artifacts from training
  data (always operates on :class:`xarray.DataArray`).
* :meth:`~VariableScaling.transform` -- apply the scaling (e.g. divide by the
  artifacts). Works on both :class:`xarray.DataArray` and in-graph tensors.
* :meth:`~VariableScaling.inverse_transform` -- reverse the scaling (e.g.
  multiply by the artifacts).
"""

from __future__ import annotations

import math
import warnings
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
_LEGACY_VARIABLE_SCALING_TYPE = "pymc_marketing.mmm.scaling.VariableScaling"
_DATA_DERIVED_SCALING_TYPE = "pymc_marketing.mmm.scaling.DataDerivedScaling"


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

    Concrete subclasses implement the ``compute`` / ``transform`` /
    ``inverse_transform`` contract:

    - :meth:`compute` -- derive scaling artifacts from training data
      (always operates on :class:`xarray.DataArray`).
    - :meth:`transform` -- apply the scaling (e.g. divide by artifacts).
      Clamps NaN/Inf results to 0. Works on both
      :class:`xarray.DataArray` and in-graph tensors.
    - :meth:`inverse_transform` -- reverse the scaling (e.g. multiply by
      artifacts).

    Parameters
    ----------
    dims : str or tuple of str
        The dimensions to perform the operation through (``"date"`` is always
        included implicitly).
    """

    dims: str | tuple[str, ...] = Field(...)

    @abstractmethod
    def scaling_description(self) -> str:
        """Human-readable summary of the scaling strategy (e.g. for logging)."""
        ...

    @abstractmethod
    def compute(self, data: xr.DataArray) -> xr.Dataset:
        """Compute scaling artifacts from training data.

        Parameters
        ----------
        data : xr.DataArray
            The raw training data for this variable.

        Returns
        -------
        xr.Dataset
            Scaling artifacts.  The dataset variable names are stable per
            subclass (e.g. ``"scale"`` for simple divisive scaling,
            ``"mean"`` and ``"std"`` for z-score).
        """
        ...

    @abstractmethod
    def transform(
        self,
        data: Any,
        artifacts: xr.Dataset,
    ) -> Any:
        """Apply the scaling.

        Clamps NaN/Inf results (e.g. from all-zero channels) to 0.

        Parameters
        ----------
        data : xr.DataArray or tensor
            The data to scale.
        artifacts : xr.Dataset
            Artifacts produced by :meth:`compute`.

        Returns
        -------
        xr.DataArray or tensor
            Scaled data, same type as *data*.
        """
        ...

    @abstractmethod
    def inverse_transform(
        self,
        data: Any,
        artifacts: xr.Dataset,
    ) -> Any:
        """Reverse the scaling.

        Parameters
        ----------
        data : xr.DataArray or tensor
            The scaled data to return to original scale.
        artifacts : xr.Dataset
            Artifacts produced by :meth:`compute`.

        Returns
        -------
        xr.DataArray or tensor
            Data in original scale, same type as *data*.
        """
        ...

    @staticmethod
    def _safe_scale(result: Any) -> Any:
        """Clamp NaN and Inf results from division to 0.

        This handles edge cases like all-zero channels (``0 / 0 → NaN``)
        or non-zero data divided by a zero scale (``nonzero / 0 → Inf``).
        Only operates on :class:`xarray.DataArray` inputs; tensor inputs
        are assumed to already contain finite values.
        """
        if isinstance(result, xr.DataArray):
            return xr.where(
                np.logical_or(np.isnan(result), np.isinf(result)),
                0.0,
                result,
            )
        return result

    @model_validator(mode="after")
    def _validate_dims(self) -> Self:
        if isinstance(self.dims, str):
            self.dims = (self.dims,)

        if "date" in self.dims:
            raise ValueError("dim 'date' is already assumed in the model.")

        if len(set(self.dims)) != len(self.dims):
            raise ValueError("dims must be unique.")

        return self


class MaxAbsScaling(VariableScaling):
    """Scale by the maximum absolute value of the data.

    Parameters
    ----------
    dims : str or tuple of str
        The dimensions to perform the operation through (``"date"`` is always
        included implicitly).

    Examples
    --------
    .. code-block:: python

        MaxAbsScaling(dims=())

    With multi-dimensional data:

    .. code-block:: python

        MaxAbsScaling(dims=("country",))
    """

    def scaling_description(self) -> str:
        """Human-readable summary of the scaling strategy."""
        return "max-absolute"

    def compute(self, data: xr.DataArray) -> xr.Dataset:
        """Compute ``max(abs(data))`` over date and configured dims."""
        reduce_dims = ("date", *self.dims)
        scale = data.max(dim=reduce_dims)
        return xr.Dataset({"scale": scale})

    def transform(
        self,
        data: Any,
        artifacts: xr.Dataset,
    ) -> Any:
        """Divide *data* by the stored scale artifact, clamping NaN/Inf to 0."""
        return self._safe_scale(data / artifacts["scale"])

    def inverse_transform(
        self,
        data: Any,
        artifacts: xr.Dataset,
    ) -> Any:
        """Multiply *data* by the stored scale artifact."""
        return data * artifacts["scale"]


class MeanAbsScaling(VariableScaling):
    """Scale by the mean absolute value of the data.

    Parameters
    ----------
    dims : str or tuple of str
        The dimensions to perform the operation through (``"date"`` is always
        included implicitly).

    Examples
    --------
    .. code-block:: python

        MeanAbsScaling(dims=())

    With multi-dimensional data:

    .. code-block:: python

        MeanAbsScaling(dims=("country",))
    """

    def scaling_description(self) -> str:
        """Human-readable summary of the scaling strategy."""
        return "mean-absolute"

    def compute(self, data: xr.DataArray) -> xr.Dataset:
        """Compute ``mean(abs(data))`` over date and configured dims."""
        reduce_dims = ("date", *self.dims)
        scale = data.mean(dim=reduce_dims)
        return xr.Dataset({"scale": scale})

    def transform(
        self,
        data: Any,
        artifacts: xr.Dataset,
    ) -> Any:
        """Divide *data* by the stored scale artifact, clamping NaN/Inf to 0."""
        return self._safe_scale(data / artifacts["scale"])

    def inverse_transform(
        self,
        data: Any,
        artifacts: xr.Dataset,
    ) -> Any:
        """Multiply *data* by the stored scale artifact."""
        return data * artifacts["scale"]


class DataDerivedScaling(VariableScaling):
    """Scale by a statistic of the data, computed at fit time.

    .. deprecated::
        Prefer the concrete subclasses :class:`MaxAbsScaling` and
        :class:`MeanAbsScaling`.  This class remains for backward
        compatibility of serialized models.

    Parameters
    ----------
    method : ``"max"`` | ``"mean"``
        The scaling method.
    dims : str or tuple of str
        The dimensions to perform the operation through (``"date"`` is always
        included implicitly).
    """

    method: Literal["max", "mean"] = Field(...)

    @model_validator(mode="after")
    def _emit_deprecation_warning(self) -> Self:
        warnings.warn(
            "DataDerivedScaling is deprecated. "
            "Use MaxAbsScaling or MeanAbsScaling instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self

    def scaling_description(self) -> str:
        """Human-readable summary of the scaling strategy."""
        return f"data-derived ({self.method})"

    def compute(self, data: xr.DataArray) -> xr.Dataset:
        """Compute scale from data using the configured method."""
        reduce_dims = ("date", *self.dims)
        scale = getattr(data, self.method)(dim=reduce_dims)
        return xr.Dataset({"scale": scale})

    def transform(
        self,
        data: Any,
        artifacts: xr.Dataset,
    ) -> Any:
        """Divide *data* by the stored scale artifact, clamping NaN/Inf to 0."""
        return self._safe_scale(data / artifacts["scale"])

    def inverse_transform(
        self,
        data: Any,
        artifacts: xr.Dataset,
    ) -> Any:
        """Multiply *data* by the stored scale artifact."""
        return data * artifacts["scale"]


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

    value: float | dict[str, float] | xr.DataArray = Field(...)

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
            fval = float(self.value)
            if math.isnan(fval) or fval <= 0:
                raise ValueError(
                    f"Fixed scaling value must be positive and non-NaN, "
                    f"got {self.value}."
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

    def compute(self, data: xr.DataArray) -> xr.Dataset:
        """Resolve the fixed value against *data* coordinate layout.

        For float values the data is ignored.  For dict and DataArray values
        the data coordinates are used to align and validate the scale.
        """
        reduce_dims = ("date", *self.dims)

        if isinstance(self.value, dict):
            scale = self._build_scale_from_dict(data, reduce_dims)
        elif isinstance(self.value, xr.DataArray):
            scale = self._align_scale_dataarray(data, self.value, reduce_dims)
        else:
            scale = xr.DataArray(float(self.value))

        return xr.Dataset({"scale": scale})

    def _build_scale_from_dict(
        self,
        data: xr.DataArray,
        reduce_dims: tuple[str, ...],
    ) -> xr.DataArray:
        value_map: dict[str, float] = self.value  # type: ignore[assignment]
        remaining_dims = [d for d in data.dims if d not in reduce_dims]
        if len(remaining_dims) != 1:
            raise ValueError(
                f"dict-valued fixed scaling requires exactly one remaining dimension "
                f"after reduction over {reduce_dims!r}; got {remaining_dims!r}. "
                f"Use an xarray.DataArray with dims {tuple(remaining_dims)!r} for "
                f"multi-dimensional fixed scales."
            )

        dim_name = remaining_dims[0]
        coords = data.coords[dim_name].values
        coord_labels = {str(c) for c in coords}
        provided_keys = set(value_map.keys())
        missing = coord_labels - provided_keys
        extra = provided_keys - coord_labels
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"missing keys: {sorted(missing)}")
            if extra:
                parts.append(f"unexpected keys: {sorted(extra)}")
            raise ValueError(
                f"Fixed scaling dict keys for dimension "
                f"'{dim_name}' do not match coordinate labels. "
                f"{'; '.join(parts)}. "
                f"Expected: {sorted(coord_labels)}."
            )

        values = np.array([value_map[str(c)] for c in coords])
        return xr.DataArray(
            values,
            dims=(dim_name,),
            coords={dim_name: coords},
        )

    def _align_scale_dataarray(
        self,
        data: xr.DataArray,
        user_scale: xr.DataArray,
        reduce_dims: tuple[str, ...],
    ) -> xr.DataArray:
        """Broadcast a user-supplied scale grid to match reduced data coordinates."""
        template = data.max(dim=reduce_dims, skipna=True).astype(float)
        zeros = xr.zeros_like(template, dtype=float)
        try:
            aligned = user_scale.astype(float) + zeros
        except (ValueError, TypeError) as e:
            raise ValueError(
                "Could not align fixed scaling DataArray with the data grid after "
                f"reduction over {reduce_dims!r}. Check dimension names and coordinate "
                f"labels. Underlying error: {e}"
            ) from e
        if np.isnan(np.asarray(aligned.values)).any():
            raise ValueError(
                "Fixed scaling DataArray produced NaNs after alignment — coordinates "
                "likely do not match the data grid on a shared dimension."
            )
        if dict(aligned.sizes) != dict(template.sizes):
            raise ValueError(
                f"Fixed scaling DataArray has shape {dict(aligned.sizes)} after "
                f"broadcast; expected {dict(template.sizes)} matching reduced data."
            )
        return aligned

    def transform(
        self,
        data: Any,
        artifacts: xr.Dataset,
    ) -> Any:
        """Divide *data* by the stored scale artifact, clamping NaN/Inf to 0."""
        return self._safe_scale(data / artifacts["scale"])

    def inverse_transform(
        self,
        data: Any,
        artifacts: xr.Dataset,
    ) -> Any:
        """Multiply *data* by the stored scale artifact."""
        return data * artifacts["scale"]


def validate_fixed_scaling_keys(
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


def deserialize_variable_scaling(d: dict[str, Any]) -> VariableScaling:
    """Deserialize a VariableScaling from a dict, handling both legacy and new formats.

    Legacy format (pre-class-split) uses a ``method`` field to discriminate.
    New format uses the ``__type__`` key injected by the serialization registry.
    """
    if "__type__" in d:
        type_key = d.get("__type__")

        # Backward compatibility: historical payloads may store the abstract
        # VariableScaling class type and rely on "method" discrimination.
        if type_key == _LEGACY_VARIABLE_SCALING_TYPE:
            method = d.get("method")
            dims = tuple(d.get("dims", ()))
            if method == "fixed":
                value = _maybe_deserialize_fixed_scaling_value(d["value"])
                return FixedScaling(dims=dims, value=value)
            if method == "max":
                return MaxAbsScaling(dims=dims)
            if method == "mean":
                return MeanAbsScaling(dims=dims)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                return DataDerivedScaling(method=method, dims=dims)

        # Backward compatibility: old DataDerivedScaling with method string
        if type_key == _DATA_DERIVED_SCALING_TYPE:
            method = d.get("method", "max")
            dims = tuple(d.get("dims", ()))
            if method == "max":
                return MaxAbsScaling(dims=dims)
            if method == "mean":
                return MeanAbsScaling(dims=dims)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                return DataDerivedScaling(method=method, dims=dims)

        return serialization.deserialize(d)

    method = d.get("method")
    dims = tuple(d.get("dims", ()))
    if method == "fixed":
        raw_value = d["value"]
        value = _maybe_deserialize_fixed_scaling_value(raw_value)
        return FixedScaling(dims=dims, value=value)
    if method == "max":
        return MaxAbsScaling(dims=dims)
    if method == "mean":
        return MeanAbsScaling(dims=dims)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
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
            target=MaxAbsScaling(dims=()),
            channel=MaxAbsScaling(dims=()),
        )

    Fixed scaling for stable production refreshes:

    .. code-block:: python

        Scaling(
            target=FixedScaling(dims=(), value=50_000.0),
            channel=FixedScaling(dims=(), value=10_000.0),
        )
    """

    target: VariableScaling = Field(...)
    channel: VariableScaling = Field(...)

    @model_validator(mode="before")
    @classmethod
    def _coerce_dict_values(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for key in ("target", "channel"):
                val = data.get(key)
                if isinstance(val, dict):
                    data[key] = deserialize_variable_scaling(val)
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
                filtered[key] = deserialize_variable_scaling(filtered[key])
        return cls.model_validate(filtered)
