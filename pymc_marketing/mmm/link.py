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
"""Link function abstraction for MMM models.

Provides the :class:`LinkFunction` enum and the :class:`LinkSpec` strategy
hierarchy that centralise all link-dependent logic (inverse link, default
likelihood, default intercept prior, target validation, and total-media
contribution graph construction).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum

import numpy as np
import pymc.dims as pmd
from pymc_extras.prior import Prior
from pytensor.xtensor import math as ptxm
from pytensor.xtensor.type import XTensorVariable


class LinkFunction(StrEnum):
    """Supported link functions for the MMM."""

    IDENTITY = "identity"
    LOG = "log"


class LinkSpec(ABC):
    """Strategy object that centralises all link-dependent behaviour.

    Subclasses implement the five link-specific decisions:

    * :meth:`inverse_link` -- map the linear predictor to the response scale.
    * :meth:`default_likelihood` -- default likelihood prior.
    * :meth:`default_intercept` -- default intercept prior.
    * :meth:`validate_target` -- fit-time target checks.
    * :meth:`create_media_contribution_deterministic` -- graph for
      ``total_media_contribution_original_scale``.
    """

    link: LinkFunction

    @abstractmethod
    def inverse_link(self, mu: XTensorVariable) -> XTensorVariable:
        """Map the linear predictor *mu* to the response scale."""

    @abstractmethod
    def default_likelihood(self, dims: tuple[str, ...]) -> Prior:
        """Return the default likelihood prior for this link."""

    @abstractmethod
    def default_intercept(self, dims: tuple[str, ...]) -> Prior:
        """Return the default intercept prior for this link."""

    @abstractmethod
    def validate_target(self, y: np.ndarray) -> None:
        """Validate that *y* is compatible with this link function.

        Raises
        ------
        ValueError
            If the target values are incompatible.
        """

    @abstractmethod
    def original_scale_transform(
        self,
        variable: XTensorVariable,
        target_scale: XTensorVariable,
    ) -> XTensorVariable:
        """Transform a model variable to the original (response) scale.

        Parameters
        ----------
        variable : XTensorVariable
            A model variable in the linear-predictor space.
        target_scale : XTensorVariable
            The target scaling factor.

        Returns
        -------
        XTensorVariable
            The variable expressed in original scale.
        """

    @abstractmethod
    def create_media_contribution_deterministic(
        self,
        mu_var: XTensorVariable,
        channel_contribution: XTensorVariable,
        target_scale: XTensorVariable,
        output_var: str = "y",
    ) -> None:
        """Register total media contribution deterministic nodes.

        Creates ``total_media_contribution_original_scale`` (and, for the log
        link, ``{output_var}_original_scale``) as :func:`pmd.Deterministic`
        nodes.
        """

    @staticmethod
    def validate_likelihood_compatibility(
        link: LinkFunction, likelihood: Prior
    ) -> None:
        """Raise if *likelihood* is incompatible with *link*.

        The identity link is compatible with any likelihood because the
        additive decomposition does not depend on the distributional form.
        The log link requires LogNormal so that the counterfactual
        decomposition (``exp(mu) - exp(mu - media)``) is correct.

        Parameters
        ----------
        link : LinkFunction
            The link function used by the model.
        likelihood : Prior
            The likelihood distribution prior.

        Raises
        ------
        ValueError
            If the combination is known to produce incorrect downstream
            decomposition or optimisation results.
        """
        if link == LinkFunction.IDENTITY:
            return

        dist_name = likelihood.distribution
        compatible = {
            LinkFunction.LOG: {"LogNormal"},
        }
        allowed = compatible.get(link, set())
        if dist_name not in allowed:
            raise ValueError(
                f"Likelihood '{dist_name}' is not compatible with link='{link.value}'. "
                f"Allowed likelihoods for link='{link.value}': {sorted(allowed)}. "
                f"Using an incompatible likelihood will produce incorrect "
                f"decomposition and optimisation results."
            )


class IdentityLinkSpec(LinkSpec):
    """Identity link: ``E[y] = mu * target_scale``."""

    link = LinkFunction.IDENTITY

    def inverse_link(self, mu: XTensorVariable) -> XTensorVariable:
        """Return *mu* unchanged (identity transform)."""
        return mu

    def default_likelihood(self, dims: tuple[str, ...]) -> Prior:
        """Return ``Normal`` likelihood prior."""
        return Prior(
            "Normal",
            sigma=Prior("HalfNormal", sigma=2, dims=dims),
            dims=("date", *dims),
        )

    def default_intercept(self, dims: tuple[str, ...]) -> Prior:
        """Return ``Normal(0, 2)`` intercept prior."""
        return Prior("Normal", mu=0, sigma=2, dims=dims)

    def validate_target(self, y: np.ndarray) -> None:
        """No-op: identity link accepts any target values."""

    def original_scale_transform(
        self,
        variable: XTensorVariable,
        target_scale: XTensorVariable,
    ) -> XTensorVariable:
        """Return ``variable * target_scale``."""
        return variable * target_scale

    def create_media_contribution_deterministic(
        self,
        mu_var: XTensorVariable,
        channel_contribution: XTensorVariable,
        target_scale: XTensorVariable,
        output_var: str = "y",
    ) -> None:
        """Register additive ``total_media_contribution_original_scale``."""
        pmd.Deterministic(
            "total_media_contribution_original_scale",
            (channel_contribution.sum(dim="date") * target_scale).sum(),
        )


class LogLinkSpec(LinkSpec):
    r"""Log link: ``E[y] = exp(mu) * target_scale``.

    When used with :class:`~pymc_marketing.mmm.components.saturation.LogSaturation`,
    the model becomes a log-log specification where coefficients have an
    elasticity-like interpretation.  Note that under the default scaling
    pipeline (``y_scaled = y / target_scale``), the intercept absorbs
    ``log(target_scale)`` and channel data is divided by ``channel_scale``,
    so the beta coefficients are **approximate** elasticities with respect
    to *scaled* spend rather than strict textbook elasticities with respect
    to raw spend.
    """

    link = LinkFunction.LOG

    def inverse_link(self, mu: XTensorVariable) -> XTensorVariable:
        """Return ``exp(mu)``."""
        return ptxm.exp(mu)

    def default_likelihood(self, dims: tuple[str, ...]) -> Prior:
        """Return ``LogNormal`` likelihood prior."""
        return Prior(
            "LogNormal",
            sigma=Prior("HalfNormal", sigma=0.5, dims=dims),
            dims=("date", *dims),
        )

    def default_intercept(self, dims: tuple[str, ...]) -> Prior:
        """Return ``Normal(0, 5)`` intercept prior (wider for log-scale)."""
        return Prior("Normal", mu=0, sigma=5, dims=dims)

    def validate_target(self, y: np.ndarray) -> None:
        """Raise ``ValueError`` if *y* contains non-positive values."""
        if np.any(y <= 0):
            raise ValueError(
                "All target values must be strictly positive when using "
                "link='log' (LogNormal likelihood). Found non-positive "
                "values in the target. Consider removing or imputing zeros/negatives."
            )

    def original_scale_transform(
        self,
        variable: XTensorVariable,
        target_scale: XTensorVariable,
    ) -> XTensorVariable:
        """Return ``exp(variable) * target_scale``."""
        return ptxm.exp(variable) * target_scale

    def create_media_contribution_deterministic(
        self,
        mu_var: XTensorVariable,
        channel_contribution: XTensorVariable,
        target_scale: XTensorVariable,
        output_var: str = "y",
    ) -> None:
        """Register counterfactual ``total_media_contribution_original_scale`` and ``{output_var}_original_scale``."""
        mu_media = channel_contribution.sum(dim="channel")
        y_hat = ptxm.exp(mu_var) * target_scale
        y_hat_no_media = ptxm.exp(mu_var - mu_media) * target_scale

        pmd.Deterministic(
            "total_media_contribution_original_scale",
            (y_hat - y_hat_no_media).sum(dim="date").sum(),
        )

        pmd.Deterministic(
            f"{output_var}_original_scale",
            y_hat.transpose("date", ...),
        )


LINK_SPECS: dict[LinkFunction, type[LinkSpec]] = {
    LinkFunction.IDENTITY: IdentityLinkSpec,
    LinkFunction.LOG: LogLinkSpec,
}


def get_link_spec(link: LinkFunction) -> LinkSpec:
    """Return the :class:`LinkSpec` instance for *link*."""
    cls = LINK_SPECS.get(link)
    if cls is None:
        raise ValueError(
            f"Unsupported link function: '{link}'. "
            f"Supported: {[lf.value for lf in LinkFunction]}"
        )
    return cls()
