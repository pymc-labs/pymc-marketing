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
"""Utility functions for PyMC-Marketing."""

import warnings
from pathlib import Path

import arviz as az


def from_netcdf(filepath: str | Path) -> az.InferenceData:
    """Load inference data from a netcdf file without `fit_data` group warnings.

    Parameters
    ----------
    filepath : str or Path
        The path to the netcdf file.

    Returns
    -------
    az.InferenceData
        The inference data.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r"fit_data group is not defined in the InferenceData scheme",
        )
        return az.from_netcdf(filepath)
