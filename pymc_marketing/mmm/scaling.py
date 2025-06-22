#   Copyright 2022 - 2025 The PyMC Labs Developers
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
"""Scaling configuration for the MMMM."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self


class VariableScaling(BaseModel):
    """How to scale a variable.

    The scaling through the dimension of 'date' is assumed and doesn't need to be specified.

    """

    method: Literal["max", "mean"] = Field(..., description="The scaling method.")
    dims: str | tuple[str, ...] = Field(
        ...,
        description="The dimensions to perform operation through.",
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


class Scaling(BaseModel):
    """Scaling configuration for the MMM.

    Examples
    --------
    Scale the target variable by max value by group of 'DMA'

    .. code-block:: python

        from pymc_marketing.mmm.multidimensional import Scaling

        scaling = Scaling(
            **{
                "target": {
                    "method": "max",
                    # Exclude 'DMA' from dims here.
                    "dims": (),
                },
            }
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
