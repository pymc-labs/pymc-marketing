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
import xarray as xr
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

    @abstractmethod
    def mean_correction(
        self,
        posterior: xr.Dataset,
        output_var: str = "y",
    ) -> xr.DataArray:
        """Per-draw factor converting median-scale outputs to the response mean.

        Counterfactual contributions are computed on the **conditional
        median** of the response (the inverse link applied to ``mu``).  For
        links whose conditional mean differs from the median, multiplying by
        this factor rescales the median-based quantity to the conditional
        mean ``E[y | mu, ...]``.

        Parameters
        ----------
        posterior : xr.Dataset
            Posterior group of the fitted model's ``InferenceData``.
        output_var : str, default ``"y"``
            Name of the observed variable, used to locate the likelihood
            scale parameter in the posterior.

        Returns
        -------
        xr.DataArray
            The multiplicative correction with ``(chain, draw, ...)`` dims
            (broadcasting over ``date``).  It is identically ``1`` for links
            whose mean equals the median (e.g. the identity link).
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

    def mean_correction(
        self,
        posterior: xr.Dataset,
        output_var: str = "y",
    ) -> xr.DataArray:
        """Return ``1`` -- for the Normal likelihood the mean equals the median."""
        return xr.DataArray(1.0)


class LogLinkSpec(LinkSpec):
    r"""Log link: ``median(y) = exp(mu) * target_scale``.

    The likelihood is ``LogNormal(mu, sigma)``, so ``exp(mu)`` is the
    conditional **median** of the response, not its mean
    (``E[y] = exp(mu + sigma**2 / 2) * target_scale``).  All predictions and
    counterfactual contributions are computed on this median scale; use the
    ``central_tendency="mean"`` option (which applies :meth:`mean_correction`,
    the ``exp(sigma**2 / 2)`` factor) to obtain mean-scale quantities.

    When used with :class:`~pymc_marketing.mmm.components.saturation.LogSaturation`,
    the model becomes a log-log specification where the coefficients have an
    elasticity interpretation.  ``LogSaturation`` requests raw (unscaled)
    channel inputs, so the elasticity is taken with respect to actual spend
    and the intercept absorbs ``log(target_scale)`` from the target scaling
    pipeline.
    """

    link = LinkFunction.LOG

    def inverse_link(self, mu: XTensorVariable) -> XTensorVariable:
        """Return ``exp(mu)`` (the conditional median of the LogNormal response)."""
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

    def mean_correction(
        self,
        posterior: xr.Dataset,
        output_var: str = "y",
    ) -> xr.DataArray:
        r"""Return ``exp(sigma**2 / 2)``, the LogNormal mean/median ratio.

        For ``y \sim \text{LogNormal}(\mu, \sigma)`` the conditional median is
        ``exp(mu)`` while the conditional mean is ``exp(mu + sigma**2 / 2)``.
        The ratio ``exp(sigma**2 / 2)`` therefore rescales a median-based
        quantity to the mean.

        Raises
        ------
        ValueError
            If the likelihood scale ``f"{output_var}_sigma"`` is not present
            in the posterior (e.g. a fixed-sigma likelihood), so the mean
            correction cannot be computed.
        """
        sigma_name = f"{output_var}_sigma"
        if sigma_name not in posterior:
            raise ValueError(
                f"Mean-scale contributions require a sampled likelihood scale "
                f"'{sigma_name}' in the posterior, which was not found. This "
                f"happens when the LogNormal sigma is fixed rather than given a "
                f"prior. Use central_tendency='median' or give sigma a prior."
            )
        return np.exp(posterior[sigma_name] ** 2 / 2)


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
