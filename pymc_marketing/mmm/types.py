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
"""Type definitions for PyMC-Marketing MMM module.

This module contains Protocol classes and type definitions used across
the MMM module for type checking and interface definitions.
"""

from typing import Any, Protocol

import pandas as pd


class MMMBuilder(Protocol):
    """Protocol for objects that can build MMM models.

    Any object passed to ``TimeSliceCrossValidator.run(mmm=...)`` must
    implement this protocol.

    Attributes
    ----------
    None

    Methods
    -------
    build_model(X, y)
        Build and return an MMM instance ready for fitting.

    See Also
    --------
    pymc_marketing.mmm.TimeSliceCrossValidator : Cross-validator that uses this protocol.

    Examples
    --------
    Create a custom builder that implements the protocol:

    >>> class MyMMMBuilder:
    ...     def __init__(self, config):
    ...         self.config = config
    ...
    ...     def build_model(self, X, y):
    ...         # Build and return an MMM instance
    ...         mmm = MMM(...)
    ...         return mmm
    >>> builder = MyMMMBuilder(config={"channels": ["tv", "radio"]})
    >>> cv = TimeSliceCrossValidator(
    ...     n_init=100, forecast_horizon=10, date_column="date"
    ... )
    >>> combined_idata = cv.run(X, y, mmm=builder)
    """

    def build_model(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Build and return an MMM instance.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix containing the date column and predictor variables.
        y : pd.Series
            Target variable.

        Returns
        -------
        object
            An MMM instance with ``fit``, ``sample_posterior_predictive``,
            and ``idata`` attributes. The returned object should be ready
            for fitting with the provided data.

        Notes
        -----
        The returned MMM instance must support the following interface:

        - ``fit(X, y, progressbar=True)``: Fit the model to data.
        - ``sample_posterior_predictive(X, ...)``: Generate predictions.
        - ``idata``: ArviZ InferenceData attribute containing posterior samples.
        - ``sampler_config``: Optional attribute for sampler configuration.
        """
        ...
