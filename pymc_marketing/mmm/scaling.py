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

from abc import ABC, abstractmethod
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from pymc_marketing.serialization import SerializableBaseModel, serialization


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
    def _abstract_guard(self) -> None: ...

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

    def _abstract_guard(self) -> None:
        pass


class FixedScaling(VariableScaling):
    """Use a user-supplied constant that stays the same across model refreshes.

    Parameters
    ----------
    dims : str or tuple of str
        The dimensions to perform the operation through (``"date"`` is always
        included implicitly).
    value : float or dict[str, float]
        Fixed scaling constant(s).  A single ``float`` applies uniformly;
        a ``dict`` maps dimension-level labels to per-level constants (useful
        for the multidimensional MMM).  All values must be positive.

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
    """

    value: float | dict[str, float] = Field(
        ...,
        description="Fixed scaling constant(s). All values must be positive.",
    )

    def _abstract_guard(self) -> None:
        pass

    @property
    def method(self) -> str:
        """Return the scaling method name."""
        return "fixed"

    @model_validator(mode="after")
    def _validate_value(self) -> Self:
        if isinstance(self.value, dict):
            for key, val in self.value.items():
                if val <= 0:
                    raise ValueError(
                        f"All fixed scaling values must be positive, "
                        f"got {val} for key '{key}'."
                    )
        elif self.value <= 0:
            raise ValueError(f"Fixed scaling value must be positive, got {self.value}.")
        return self


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
        return FixedScaling(dims=dims, value=d["value"])
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
                if isinstance(val, dict) and not isinstance(val, VariableScaling):
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
