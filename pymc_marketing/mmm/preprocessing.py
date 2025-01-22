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
"""Preprocessing methods for the Marketing Mix Model."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

__all__ = [
    "MaxAbsScaleChannels",
    "MaxAbsScaleTarget",
    "StandardizeControls",
    "preprocessing_method_X",
    "preprocessing_method_y",
]


def preprocessing_method_X(method: Callable) -> Callable:
    """Tag a method as a preprocessing method for the X data.

    Decorator to mark a method as a preprocessing method for the X data.

    Parameters
    ----------
    method : Callable
        The method to tag as a preprocessing method for the X data.

    Returns
    -------
    Callable
        The tagged method.

    """
    if not hasattr(method, "_tags"):
        method._tags = {}  # type: ignore
    method._tags["preprocessing_X"] = True  # type: ignore
    return method


def preprocessing_method_y(method: Callable) -> Callable:
    """Tag a method as a preprocessing method for the y data.

    Decorator to mark a method as a preprocessing method for the y data.

    Parameters
    ----------
    method : Callable
        The method to tag as a preprocessing method for the y data.

    Returns
    -------
    Callable
        The tagged method.

    """
    if not hasattr(method, "_tags"):
        method._tags = {}  # type: ignore
    method._tags["preprocessing_y"] = True  # type: ignore
    return method


class MaxAbsScaleTarget:
    """MaxAbsScaler for the target data."""

    target_transformer: Pipeline

    @preprocessing_method_y
    def max_abs_scale_target_data(
        self, data: pd.Series | np.ndarray
    ) -> np.ndarray | pd.Series:
        """MaxAbsScaler for the target data.

        Parameters
        ----------
        data : pd.Series | np.ndarray
            The target data to scale.

        Returns
        -------
        np.ndarray | pd.Series
            The scaled target data.

        """
        if isinstance(data, pd.Series):
            data = data.to_numpy()

        target_vector = data.reshape(-1, 1)
        transformers = [("scaler", MaxAbsScaler())]
        pipeline = Pipeline(steps=transformers)
        self.target_transformer: Pipeline = pipeline.fit(X=target_vector)
        data = self.target_transformer.transform(X=target_vector).flatten()
        return data


class MaxAbsScaleChannels:
    """MaxAbsScaler for the channel data."""

    channel_columns: list[str] | tuple[str]

    @preprocessing_method_X
    def max_abs_scale_channel_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """MaxAbsScaler for the channel data.

        Parameters
        ----------
        data : pd.DataFrame
            The channel data to scale.

        Returns
        -------
        pd.DataFrame
            The scaled channel data.

        """
        data_cp = data.copy()
        channel_data: pd.DataFrame | pd.Series[Any] = data_cp[self.channel_columns]
        transformers = [("scaler", MaxAbsScaler())]
        pipeline: Pipeline = Pipeline(steps=transformers)
        self.channel_transformer: Pipeline = pipeline.fit(X=channel_data.to_numpy())
        data_cp[self.channel_columns] = self.channel_transformer.transform(
            channel_data.to_numpy()
        )
        return data_cp


class StandardizeControls:
    """StandardScaler for the control data."""

    control_columns: list[str]  # TODO: Handle Optional[List[str]]

    @preprocessing_method_X
    def standardize_control_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """StandardScaler for the control data.

        Parameters
        ----------
        data : pd.DataFrame
            The control data to scale.

        Returns
        -------
        pd.DataFrame
            The scaled control data.

        """
        control_data: pd.DataFrame = data[self.control_columns]
        transformers = [("scaler", StandardScaler())]
        pipeline: Pipeline = Pipeline(steps=transformers)
        self.control_transformer: Pipeline = pipeline.fit(X=control_data.to_numpy())
        data[self.control_columns] = self.control_transformer.transform(
            control_data.to_numpy()
        )
        return data
