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

"""Graph evaluation for posterior experiment designs."""

from __future__ import annotations

from collections.abc import Callable
from functools import cached_property
from typing import TYPE_CHECKING, cast

import numpy as np
import pytensor.xtensor as ptx
from pytensor import function
from pytensor.graph.replace import clone_replace
from pytensor.xtensor.vectorization import vectorize_graph

from pymc_marketing.mmm.link import _distribution_name
from pymc_marketing.names.mmm import Dim, Var
from pymc_marketing.pytensor_utils import extract_response_distribution

if TYPE_CHECKING:
    from pymc import Model
    from xarray import Dataset

    from pymc_marketing.mmm.mmm import MMM


def validate_observed_mean(mmm: MMM, posterior: Dataset) -> None:
    """Require the predictor to give a valid observed mean for any spend change."""
    likelihood = mmm.model_config["likelihood"]
    distribution = _distribution_name(likelihood)
    # A go-dark intervention can leave a negative predictor although the
    # fitted predictor was positive. Require a likelihood defined for any mu.
    if distribution not in {"Normal", "Laplace", "StudentT"}:
        raise ValueError(
            "ExperimentDesigner requires a likelihood with a defined conditional mean "
            "equal to mu for any real mu. Supported likelihoods are Normal, Laplace, "
            f"and StudentT with a defined mean. Got {distribution!r}."
        )
    try:
        # The link validator rejects wrappers and StudentT without a mean.
        mmm._link_spec.mean_scale_factor(posterior, likelihood, mmm.output_var)
    except ValueError as error:
        raise ValueError(
            f"ExperimentDesigner requires a defined conditional mean equal to mu. "
            f"The fitted {distribution!r} likelihood does not meet this requirement."
        ) from error


class ContributionGraph:
    """Evaluate a channel response graph with posterior values bound."""

    def __init__(self, model: Model, posterior: Dataset, n_channels: int) -> None:
        observed_spend = model[Var.CHANNEL_DATA]
        self._spend = ptx.xtensor(
            name="channel_data_input",
            dtype=observed_spend.dtype,
            shape=(None, n_channels),
            dims=(Dim.DATE, Dim.CHANNEL),
        )
        response = extract_response_distribution(
            model, posterior, Var.CHANNEL_CONTRIBUTION
        )
        [graph] = clone_replace([response], replace={observed_spend: self._spend})
        response = cast("ptx.XTensorVariable", graph)
        if "sample" not in response.dims:
            # Fixed media parameters have the same response in every draw.
            sizes = posterior.sizes
            draws = np.ones(sizes["chain"] * sizes["draw"], dtype=response.dtype)
            response = response * ptx.as_xtensor(draws, dims=("sample",))
        self._response = response.transpose("sample", Dim.DATE, Dim.CHANNEL)

    @cached_property
    def _evaluate(self) -> Callable[[np.ndarray], np.ndarray]:
        return function([self._spend], self._response)

    @cached_property
    def _evaluate_sweep(self) -> Callable[[np.ndarray], np.ndarray]:
        sweep = ptx.xtensor(
            name="spend_sweep",
            dtype=self._spend.dtype,
            dims=(Dim.SWEEP, Dim.DATE, Dim.CHANNEL),
        )
        response = self._response.isel(**{Dim.DATE: -1})
        response = vectorize_graph(response, replace={self._spend: sweep})
        return function([sweep], response.transpose("sample", Dim.SWEEP, Dim.CHANNEL))

    def __call__(self, spend: np.ndarray) -> np.ndarray:
        return np.asarray(self._evaluate(spend.astype(self._spend.dtype, copy=False)))

    def steady_responses(self, spend: np.ndarray) -> np.ndarray:
        """Return final-period response for each draw, spend scenario and channel."""
        return np.asarray(
            self._evaluate_sweep(spend.astype(self._spend.dtype, copy=False))
        )
