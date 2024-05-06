#   Copyright 2024 The PyMC Labs Developers
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
"""Validating methods for MMM classes."""

from collections.abc import Callable

import pandas as pd

__all__ = [
    "validation_method_X",
    "validation_method_y",
    "ValidateControlColumns",
    "ValidateTargetColumn",
    "ValidateDateColumn",
    "ValidateChannelColumns",
]


def validation_method_y(method: Callable) -> Callable:
    if not hasattr(method, "_tags"):
        method._tags = {}  # type: ignore
    method._tags["validation_y"] = True  # type: ignore
    return method


def validation_method_X(method: Callable) -> Callable:
    if not hasattr(method, "_tags"):
        method._tags = {}  # type: ignore
    method._tags["validation_X"] = True  # type: ignore
    return method


class ValidateTargetColumn:
    @validation_method_y
    def validate_target(self, data: pd.Series) -> None:
        if len(data) == 0:
            raise ValueError("y must have at least one element")


class ValidateDateColumn:
    date_column: str

    @validation_method_X
    def validate_date_col(self, data: pd.DataFrame) -> None:
        if self.date_column not in data.columns:
            raise ValueError(f"date_col {self.date_column} not in data")
        if not data[self.date_column].is_unique:
            raise ValueError(f"date_col {self.date_column} has repeated values")


class ValidateChannelColumns:
    channel_columns: list[str] | tuple[str]

    @validation_method_X
    def validate_channel_columns(self, data: pd.DataFrame) -> None:
        if not isinstance(self.channel_columns, list | tuple):
            raise ValueError("channel_columns must be a list or tuple")
        if len(self.channel_columns) == 0:
            raise ValueError("channel_columns must not be empty")
        if not set(self.channel_columns).issubset(data.columns):
            raise ValueError(f"channel_columns {self.channel_columns} not in data")
        if len(set(self.channel_columns)) != len(self.channel_columns):
            raise ValueError(
                f"channel_columns {self.channel_columns} contains duplicates"
            )
        if (data.filter(list(self.channel_columns)) < 0).any().any():
            raise ValueError(
                f"channel_columns {self.channel_columns} contains negative values"
            )


class ValidateControlColumns:
    control_columns: list[str] | None

    @validation_method_X
    def validate_control_columns(self, data: pd.DataFrame) -> None:
        if self.control_columns is None:
            return None
        if not isinstance(self.control_columns, list | tuple):
            raise ValueError("control_columns must be None, a list or tuple")
        if len(self.control_columns) == 0:
            raise ValueError(
                "If control_columns is not None, then it must not be empty"
            )
        if not set(self.control_columns).issubset(data.columns):
            raise ValueError(f"control_columns {self.control_columns} not in data")
        if len(set(self.control_columns)) != len(self.control_columns):
            raise ValueError(
                f"control_columns {self.control_columns} contains duplicates"
            )
