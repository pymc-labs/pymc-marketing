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

from typing import Literal, Self

from pydantic import Field, model_validator

from pymc_marketing.serialization import SerializableBaseModel


class VariableScaling(SerializableBaseModel):
    """How to scale a variable.

    Supported methods:

    - ``"max"``: scale by the absolute maximum of the data (computed at fit time).
    - ``"mean"``: scale by the absolute mean of the data (computed at fit time).
    - ``"fixed"``: use a user-supplied constant that stays the same across model
      refreshes.  Requires the ``value`` parameter.

    The scaling through the dimension of ``'date'`` is assumed and doesn't need
    to be specified.

    Parameters
    ----------
    method : ``"max"`` | ``"mean"`` | ``"fixed"``
        The scaling method.
    dims : str or tuple of str
        The dimensions to perform the operation through (``"date"`` is always
        included implicitly).
    value : float or dict[str, float] or None
        Fixed scaling constant(s).  Required when ``method="fixed"``, forbidden
        otherwise.  A single ``float`` applies uniformly; a ``dict`` maps
        dimension-level labels to per-level constants (useful for the
        multidimensional MMM).  All values must be positive.

    Examples
    --------
    Data-derived scaling (default behaviour):

    .. code-block:: python

        VariableScaling(method="max", dims=())

    Fixed scalar scaling for production stability:

    .. code-block:: python

        VariableScaling(method="fixed", dims=(), value=10_000.0)

    Per-dimension fixed scaling (multidimensional MMM):

    .. code-block:: python

        VariableScaling(
            method="fixed",
            dims=("country",),
            value={"US": 50_000, "UK": 30_000},
        )
    """

    method: Literal["max", "mean", "fixed"] = Field(
        ..., description="The scaling method."
    )
    dims: str | tuple[str, ...] = Field(
        ...,
        description="The dimensions to perform operation through.",
    )
    value: float | dict[str, float] | None = Field(
        default=None,
        description=(
            "Fixed scaling constant(s). Required when method='fixed', "
            "forbidden otherwise."
        ),
    )

    @model_validator(mode="after")
    def _validate_dims(self) -> Self:
        if isinstance(self.dims, str):
            self.dims = (self.dims,)

        if "date" in self.dims:
            raise ValueError("dim of 'date' of is already assumed in the model.")

        if len(set(self.dims)) != len(self.dims):
            raise ValueError("dims must be unique.")

        return self

    @model_validator(mode="after")
    def _validate_fixed_value(self) -> Self:
        if self.method == "fixed":
            if self.value is None:
                raise ValueError("value is required when method='fixed'.")
            if isinstance(self.value, dict):
                for key, val in self.value.items():
                    if val <= 0:
                        raise ValueError(
                            f"All fixed scaling values must be positive, "
                            f"got {val} for key '{key}'."
                        )
            elif self.value <= 0:
                raise ValueError(
                    f"Fixed scaling value must be positive, got {self.value}."
                )
        else:
            if self.value is not None:
                raise ValueError(
                    f"value must be None when method='{self.method}', got {self.value}."
                )
        return self


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
            target=VariableScaling(method="max", dims=()),
            channel=VariableScaling(method="max", dims=()),
        )

    Fixed scaling for stable production refreshes:

    .. code-block:: python

        Scaling(
            target=VariableScaling(method="fixed", dims=(), value=50_000.0),
            channel=VariableScaling(method="fixed", dims=(), value=10_000.0),
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
