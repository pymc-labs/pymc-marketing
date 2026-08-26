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

import warnings
from abc import ABC, abstractmethod
from enum import StrEnum

import numpy as np
import pymc.dims as pmd
import xarray as xr
from pymc_extras.prior import Prior
from pytensor.xtensor import math as ptxm
from pytensor.xtensor.type import XTensorVariable
from scipy.special import erfcx
from scipy.stats import truncnorm


class LinkFunction(StrEnum):
    """Supported link functions for the MMM."""

    IDENTITY = "identity"
    LOG = "log"


#: Likelihoods whose ``mu`` parameter is on the scale of the response, so the
#: additive decomposition under the identity link is in the units of the target.
#: This is about units only.  ``mu`` still need not equal ``E[y]``: under
#: ``TruncatedNormal`` it does not, so ``*_original_scale`` will not reconcile
#: against the posterior predictive mean.  See issue #2834.
#: Three ``pymc.dims`` likelihoods take ``mu`` on the response scale and are
#: still left out.  ``Poisson`` and ``NegativeBinomial`` are discrete while the
#: likelihood is observed on the target divided by ``target_scale``, which is
#: not integer-valued, so they cannot be used under this model at all.
#: ``Beta`` needs the target inside ``(0, 1)``, which the scaling does not
#: guarantee, and nothing checks the target against the likelihood support yet.
#: See issue #2835.
RESPONSE_SCALE_LIKELIHOODS = frozenset(
    {
        "Normal",
        "StudentT",
        "TruncatedNormal",
        "Gamma",
        "Laplace",
        "InverseGamma",
    }
)

#: Likelihoods whose ``mu`` parameter is on some other scale, mapped to the name
#: of that scale.  Rejected under the identity link.
NON_RESPONSE_SCALE_LIKELIHOODS = {"LogNormal": "log"}

#: Likelihoods allowed for the non-identity links, which each need one specific
#: distributional form for their counterfactual decomposition to be correct.
LINK_LIKELIHOODS = {LinkFunction.LOG: frozenset({"LogNormal"})}


#: Likelihoods whose identity-link mean correction is an offset rather than a
#: factor, so it cannot be folded into a scale.
ADDITIVE_CORRECTION_LIKELIHOODS = frozenset({"TruncatedNormal"})

#: Key of the baseline term in a counterfactual contribution dataset.  Under
#: the identity link a correction that belongs to the noise distribution is
#: added here rather than spread across the component contributions.
BASELINE_PART = "intercept"


def _check_studentt_mean_exists(
    posterior: xr.Dataset,
    likelihood: Prior,
    output_var: str,
) -> None:
    """Raise if the StudentT degrees of freedom leave the mean undefined.

    ``E[y]`` exists only for ``nu > 1``.  ``nu`` may be fixed or sampled, so
    check whichever applies.  A single draw at or below 1 is enough: ``E[y]``
    does not exist for that draw, so the posterior of the mean-scale
    contributions has a hole in it and cannot be summarised.  The message
    reports how many draws are affected, since one stray draw and a posterior
    concentrated below 1 need different fixes.
    """
    nu = likelihood.parameters.get("nu")
    nu_name = f"{output_var}_nu"
    share = ""

    if isinstance(nu, Prior):
        # Sampled but absent from the posterior: nothing to check against.
        # This is the pre-sampling path, where the caller has no draws yet.
        if nu_name not in posterior:
            return
        nu_values = posterior[nu_name]
        smallest = float(nu_values.min())
        offending = int((nu_values <= 1).sum())
        share = f" {offending} of {nu_values.size} draws are at or below 1, and the"
    elif nu is None:
        return
    else:
        smallest = float(np.min(nu))
        share = " The"

    if smallest <= 1:
        raise ValueError(
            f"A StudentT likelihood has no mean when nu <= 1, so mean-scale "
            f"contributions are undefined.{share} smallest value found is "
            f"{smallest:.4g}. Use central_tendency='median', or keep nu above 1."
        )


def _distribution_name(likelihood: Prior) -> str:
    """Return the distribution name of *likelihood*.

    Wrappers such as ``Censored`` hold another prior in ``distribution``
    instead of a name, so unwrap until a name is reached.  Objects without a
    ``distribution`` at all, such as the ``SpecialPrior`` subclasses, fall
    back to their class name, so the checks below compare a real name rather
    than ``None``.
    """
    dist = getattr(likelihood, "distribution", None)
    while dist is not None and not isinstance(dist, str):
        dist = getattr(dist, "distribution", None)
    return dist if dist is not None else type(likelihood).__name__


class LinkSpec(ABC):
    """Strategy object that centralises all link-dependent behaviour.

    Subclasses implement the five link-specific decisions:

    * :meth:`inverse_link` -- map the linear predictor to the response scale.
    * :meth:`default_likelihood` -- default likelihood prior.
    * :meth:`default_intercept` -- default intercept prior.
    * :meth:`validate_target` -- fit-time target checks.
    * :meth:`create_media_contribution_deterministic` -- graph for
      ``total_media_contribution_original_scale``.

    One concrete helper is shared by all links:
    :meth:`create_total_response_deterministic` (the mu-effect objective
    ``total_response_original_scale``, registered by ``MMM.build_model`` only
    when the model has mu effects).
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

    def create_total_response_deterministic(
        self,
        mu_var: XTensorVariable,
        target_scale: XTensorVariable,
    ) -> None:
        """Register ``total_response_original_scale``.

        The total predicted response (original scale, scalar per draw),
        computed via :meth:`original_scale_transform` so it is correct for
        every link.  Because ``mu_var`` already includes every additive
        mu-effect, this is the natural objective for optimizing an effect's
        lever, or a mediated funnel path, jointly with media
        (:class:`~pymc_marketing.mmm.budget_optimizer.BudgetOptimizer` with
        ``response_variable="total_response_original_scale"``).

        The result is a scalar: the sum reduces **every** dimension, so a model
        with extra dims (geo, product) totals across all of them. That is the
        right contract for a single shared budget, and the wrong one if segments
        hold separate budgets -- those want a per-segment objective and a
        constraint per segment.

        Parameters
        ----------
        mu_var : XTensorVariable
            The finalized linear predictor, including every mu effect.
        target_scale : XTensorVariable
            The target scaling factor.

        Warnings
        --------
        Unlike ``total_media_contribution_original_scale``, this quantity
        includes the (approximately constant) baseline response.  For the
        default mean utility the ``argmax`` is unchanged, but a risk-adjusted
        utility function shifts the mean/variance trade-off, so those should
        prefer a media or effect contribution response variable.  In an
        optimization model the sum also runs over the full date coord, which
        includes the ``adstock_periods`` carry-over tail, so an event window
        landing in that tail would be optimized against periods outside the
        intended plan.

        This is a response total, not a media attribution: for the
        direct-versus-mediated decomposition see
        :class:`~pymc_marketing.mmm.incrementality.Incrementality`, which
        computes proper counterfactuals.

        Under the log link this sums ``exp(mu) * target_scale``, the conditional
        *median* rather than the mean; :meth:`mean_scale_factor` is the factor
        between them. The argmax is unaffected, since that factor is per-draw
        and budget-independent, but a reader taking the value itself as the
        expected response is off by it.
        """
        pmd.Deterministic(
            "total_response_original_scale",
            self.original_scale_transform(mu_var, target_scale).sum(),
        )

    @abstractmethod
    def to_mean_scale(
        self,
        dataset: xr.Dataset,
        posterior: xr.Dataset,
        likelihood: Prior,
        target_scale: xr.DataArray,
        output_var: str = "y",
    ) -> xr.Dataset:
        """Rescale a median-scale contribution dataset to the response mean.

        Counterfactual contributions are computed on the **conditional
        median** of the response (the inverse link applied to ``mu``).  Where
        the conditional mean differs from the median, this applies the
        correction and returns the mean-scale dataset.

        The correction's *form* is link-dependent, which is why this applies
        it rather than returning a factor.  Under the log link the model is
        multiplicative in the components, so a proportional factor is right.
        Under the identity link the discrepancy belongs to the noise
        distribution rather than to any component, so it is added to the
        baseline term instead of being spread across all of them.

        Parameters
        ----------
        dataset : xr.Dataset
            Median-scale counterfactual contributions, one variable per
            component, including an ``"intercept"`` baseline term.
        posterior : xr.Dataset
            Posterior group of the fitted model's ``DataTree``.
        likelihood : Prior
            The likelihood prior.  Under the identity link the correction
            depends on it, not only on the link.
        target_scale : xr.DataArray
            The target scaling factor.  ``dataset`` is in target units while
            the likelihood parameters are on the scaled axis.
        output_var : str, default ``"y"``
            Name of the observed variable, used to locate the likelihood
            parameters in the posterior.

        Returns
        -------
        xr.Dataset
            The dataset on the conditional-mean scale.
        """

    def mean_correction(
        self,
        posterior: xr.Dataset,
        output_var: str = "y",
    ) -> xr.DataArray:
        """Per-draw factor converting median-scale outputs to the response mean.

        .. deprecated:: 1.1.0
            Use :meth:`to_mean_scale`, or :meth:`mean_scale_factor` where a
            factor is what the caller needs.  A single multiplicative factor
            cannot express the identity-link correction, which depends on the
            likelihood and is additive rather than proportional.

        Parameters
        ----------
        posterior : xr.Dataset
            Posterior group of the fitted model's ``DataTree``.
        output_var : str, default ``"y"``
            Name of the observed variable, used to locate the likelihood
            scale parameter in the posterior.

        Returns
        -------
        xr.DataArray
            The multiplicative correction with ``(chain, draw, ...)`` dims
            (broadcasting over ``date``).

        Raises
        ------
        ValueError
            Under the identity link, where the right correction depends on the
            likelihood this signature cannot see.  See
            :meth:`IdentityLinkSpec.mean_correction`.
        """
        warnings.warn(
            f"{type(self).__name__}.mean_correction is deprecated, use "
            f"to_mean_scale instead. A single factor cannot express the "
            f"identity-link correction, which depends on the likelihood and "
            f"is additive rather than proportional.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._mean_ratio(posterior, output_var)

    def mean_scale_factor(
        self,
        posterior: xr.Dataset,
        likelihood: Prior,
        output_var: str = "y",
    ) -> xr.DataArray:
        """Return the mean correction as a multiplicative factor.

        For callers that fold the correction into a scale rather than applying
        it to a contribution dataset.  Raises where the correction is not
        expressible as a factor, which is the whole reason
        :meth:`to_mean_scale` exists.

        Parameters
        ----------
        posterior : xr.Dataset
            Posterior group of the fitted model's ``DataTree``.
        likelihood : Prior
            The likelihood prior.  Under the identity link the correction
            depends on it, not only on the link.
        output_var : str, default ``"y"``
            Name of the observed variable, used to locate the likelihood
            parameters in the posterior.

        Returns
        -------
        xr.DataArray
            The multiplicative correction with ``(chain, draw, ...)`` dims
            (broadcasting over ``date``).

        Raises
        ------
        ValueError
            If ``E[y]`` is undefined for *likelihood*, or if the correction for
            this link and likelihood is additive.

        Warns
        -----
        UserWarning
            If no correction is known for *likelihood*, in which case the
            factor is ``1``.
        """
        # Shared with to_mean_scale so both entry points reject and warn about
        # the same likelihoods. as_factor lets the link that has an additive
        # correction refuse it here, rather than the base class testing which
        # subclass it is.
        self._validate_mean_defined(posterior, likelihood, output_var, as_factor=True)
        return self._mean_ratio(posterior, output_var)

    def _validate_mean_defined(
        self,
        posterior: xr.Dataset,
        likelihood: Prior,
        output_var: str,
        as_factor: bool = False,
    ) -> None:
        """Raise or warn where ``E[y]`` is undefined or unknown for *likelihood*.

        No-op by default: :meth:`validate_likelihood_compatibility` pins each
        non-identity link to one likelihood, so there is nothing left to
        dispatch on.  ``IdentityLinkSpec`` overrides it.

        Parameters
        ----------
        posterior : xr.Dataset
            Posterior group of the fitted model's ``DataTree``.
        likelihood : Prior
            The likelihood prior to check.
        output_var : str
            Name of the observed variable.
        as_factor : bool, default ``False``
            Whether the caller needs the correction as a multiplicative
            factor, which a link whose correction is additive must refuse.
        """
        return None

    def _mean_ratio(
        self,
        posterior: xr.Dataset,
        output_var: str = "y",
    ) -> xr.DataArray:
        """Return the mean/median ratio, ``1`` unless a link overrides it."""
        return xr.DataArray(1.0)

    @staticmethod
    def validate_likelihood_compatibility(
        link: LinkFunction, likelihood: Prior
    ) -> None:
        """Raise if *likelihood* is incompatible with *link*.

        The criterion is whether ``mu`` lives on the scale of the response.
        Under the identity link every contribution Deterministic is a share of
        ``mu``, so a likelihood that places ``mu`` on another scale (LogNormal
        places it on the log scale) turns each ``*_original_scale`` variable
        into a delta on that other scale multiplied by ``target_scale``, which
        is not a contribution in any units.  Likelihoods that are not
        recognised warn instead of raising, so custom priors keep building.

        The log link requires LogNormal so that the counterfactual
        decomposition (``exp(mu) - exp(mu - media)``) is correct.

        The error message tells the reader to flip ``link`` and rebuild rather
        than refit.  That works because the likelihood is handed the linear
        predictor directly and ``inverse_link`` is never applied to it, so
        ``link='identity'`` and ``link='log'`` with the same likelihood give
        the same observed-variable graph and the same free variables.  Only
        the ``*_original_scale`` Deterministics differ, so an existing
        posterior is reinterpreted rather than invalidated.

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

        Warns
        -----
        UserWarning
            If the likelihood is not one whose ``mu`` scale is known.
        """
        dist_name = _distribution_name(likelihood)

        if link == LinkFunction.IDENTITY:
            if dist_name in NON_RESPONSE_SCALE_LIKELIHOODS:
                scale = NON_RESPONSE_SCALE_LIKELIHOODS[dist_name]
                raise ValueError(
                    f"Likelihood '{dist_name}' is not compatible with "
                    f"link='identity'. Its 'mu' is on the {scale} scale, not on "
                    f"the scale of the target, so every '*_original_scale' "
                    f"contribution would be a {scale}-scale delta multiplied by "
                    "'target_scale'. Use link='log' with LogNormal (it needs a "
                    "strictly positive target), or keep link='identity' with a "
                    "likelihood whose 'mu' is the response scale: "
                    f"{sorted(RESPONSE_SCALE_LIKELIHOODS)}. "
                    "To repair an already saved model without refitting:\n"
                    "    kwargs = MMM.idata_to_init_kwargs(idata)\n"
                    "    kwargs['link'] = 'log'  # or edit "
                    "kwargs['model_config']['likelihood']\n"
                    "    mmm = MMM(**kwargs)"
                )
            if dist_name not in RESPONSE_SCALE_LIKELIHOODS:
                warnings.warn(
                    f"Likelihood '{dist_name}' is "
                    "not a known response-scale likelihood. With "
                    "link='identity' the contribution decomposition assumes "
                    "'mu' is on the scale of the target. Check that it is "
                    "before reading '*_original_scale' variables. Known "
                    "response-scale likelihoods: "
                    f"{sorted(RESPONSE_SCALE_LIKELIHOODS)}.",
                    UserWarning,
                    stacklevel=2,
                )
            return

        allowed = LINK_LIKELIHOODS.get(link, frozenset())
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

    def to_mean_scale(
        self,
        dataset: xr.Dataset,
        posterior: xr.Dataset,
        likelihood: Prior,
        target_scale: xr.DataArray,
        output_var: str = "y",
    ) -> xr.Dataset:
        """Apply the identity-link mean correction, which depends on *likelihood*.

        ``mu`` is in the units of the target, so most response-scale
        likelihoods have ``E[y] == mu`` and nothing to do.  ``TruncatedNormal``
        is the exception: clipping shifts the mean off ``mu`` by an amount that
        belongs to the noise distribution, not to any component of the linear
        predictor.  The offset is therefore added to the baseline term and the
        component contributions are left alone.

        Raises
        ------
        ValueError
            If no mean correction is defined for *likelihood*, or if
            ``TruncatedNormal`` needs the baseline term and *dataset* has none.

        Warns
        -----
        UserWarning
            If no correction is known for *likelihood*, in which case *dataset*
            is returned unchanged, on the median scale.
        """
        self._validate_mean_defined(posterior, likelihood, output_var)

        if _distribution_name(likelihood) == "TruncatedNormal":
            if BASELINE_PART not in dataset:
                raise ValueError(
                    f"The truncation correction is added to the "
                    f"'{BASELINE_PART}' term, which is missing from the "
                    f"contribution dataset (found "
                    f"{sorted(dataset.data_vars)}). Use "
                    f"central_tendency='median'."
                )
            offset = self._truncation_offset(posterior, likelihood, output_var)
            corrected = dataset.copy()
            corrected[BASELINE_PART] = corrected[BASELINE_PART] + offset * target_scale
            return corrected

        # Everything the validator let through has E[y] == mu, so the
        # median-scale dataset is already on the mean scale.
        return dataset

    def _validate_mean_defined(
        self,
        posterior: xr.Dataset,
        likelihood: Prior,
        output_var: str,
        as_factor: bool = False,
    ) -> None:
        """Reject the identity-link likelihoods whose ``E[y]`` is not ``mu``.

        Both :meth:`to_mean_scale` and :meth:`mean_scale_factor` go through
        here, so a likelihood that cannot be corrected is caught whichever
        entry point the caller uses.

        Parameters
        ----------
        posterior : xr.Dataset
            Posterior group of the fitted model's ``DataTree``.
        likelihood : Prior
            The likelihood prior to check.
        output_var : str
            Name of the observed variable.
        as_factor : bool, default ``False``
            Whether the caller needs a multiplicative factor, which the
            likelihoods with an additive correction cannot provide.

        Raises
        ------
        ValueError
            If *likelihood* is a wrapper, has ``mu`` off the response scale,
            has no mean, or (when *as_factor*) needs an additive correction.

        Warns
        -----
        UserWarning
            If no correction is known for *likelihood*.
        """
        dist_name = _distribution_name(likelihood)

        # Wrappers such as Censored and Scaled resolve to the name of the
        # distribution they hold, but move its mean, so E[y] != mu even for the
        # response-scale names. Reject them before dispatching, rather than
        # silently returning median-scale numbers labelled as means. They are
        # told apart by holding no parameters of their own.
        if getattr(likelihood, "parameters", None) is None:
            raise ValueError(
                f"No mean correction is defined for a wrapped likelihood "
                f"({type(likelihood).__name__} holding '{dist_name}'). The "
                f"wrapper moves the mean off 'mu', so the contributions cannot "
                f"be read as means. Use central_tendency='median'."
            )

        # validate_likelihood_compatibility rejects these at build time, so a
        # model cannot reach here with one. Rejected again rather than warned
        # about, because 'mu' is not even in the units of the target.
        if dist_name in NON_RESPONSE_SCALE_LIKELIHOODS:
            scale = NON_RESPONSE_SCALE_LIKELIHOODS[dist_name]
            raise ValueError(
                f"Likelihood '{dist_name}' has 'mu' on the {scale} scale, not "
                f"on the scale of the target, so there is no mean correction "
                f"for it under link='identity'. Use link='log', or "
                f"central_tendency='median'."
            )

        if as_factor and dist_name in ADDITIVE_CORRECTION_LIKELIHOODS:
            raise ValueError(
                f"The mean correction for '{dist_name}' under link='identity' "
                f"is an offset, not a factor, so it cannot be folded into a "
                f"scale. Use central_tendency='median'."
            )

        if dist_name == "StudentT":
            _check_studentt_mean_exists(posterior, likelihood, output_var)
            return None

        if dist_name in RESPONSE_SCALE_LIKELIHOODS:
            return None

        warnings.warn(
            f"No mean correction is known for likelihood '{dist_name}' under "
            f"link='identity', so the contributions are returned on the median "
            f"scale. Check whether E[y] equals 'mu' for it before reading them "
            f"as means.",
            UserWarning,
            # This runs one frame below the public entry point, so 3 lands on
            # the caller of to_mean_scale / mean_scale_factor.
            stacklevel=3,
        )
        return None

    def mean_correction(
        self,
        posterior: xr.Dataset,
        output_var: str = "y",
    ) -> xr.DataArray:
        """Refuse: the identity-link correction needs the likelihood.

        .. deprecated:: 1.1.0
            Use :meth:`to_mean_scale`, or :meth:`mean_scale_factor` where a
            factor is what the caller needs.

        The base implementation returns ``1``, which was the legacy behaviour
        and is wrong for ``TruncatedNormal``.  Rather than keep returning a
        known-wrong number from a public method, this refuses and names the
        replacements.  Which correction applies is decided by the likelihood,
        and this signature cannot see it.

        Parameters
        ----------
        posterior : xr.Dataset
            Unused; kept for the inherited signature.
        output_var : str, default ``"y"``
            Unused; kept for the inherited signature.

        Raises
        ------
        ValueError
            Always.
        """
        raise ValueError(
            "IdentityLinkSpec.mean_correction is deprecated and no longer "
            "returns a value: under link='identity' the right correction "
            "depends on the likelihood, which this signature cannot see, and "
            "returning 1 is wrong for TruncatedNormal. Use to_mean_scale to "
            "correct a contribution dataset, or mean_scale_factor to get a "
            "factor."
        )

    @staticmethod
    def _truncation_offset(
        posterior: xr.Dataset,
        likelihood: Prior,
        output_var: str,
    ) -> xr.DataArray:
        r"""Return ``E[y] - mu`` for a truncated Normal, per draw and date.

        For ``y`` truncated to ``[lower, upper]`` with ``alpha = (lower - mu)
        / sigma`` and ``beta = (upper - mu) / sigma``:

        .. math::

            E[y] - \mu = \sigma \,
                \frac{\phi(\alpha) - \phi(\beta)}{\Phi(\beta) - \Phi(\alpha)}

        One-sided truncation is evaluated through ``erfcx`` rather than that
        expression directly, since the ratio loses all precision far from the
        bound.  The value is the same.
        """
        if "mu" not in posterior:
            raise ValueError(
                "Mean-scale contributions under link='identity' with a "
                "TruncatedNormal likelihood need 'mu' in the posterior, which "
                "was not found. Models fitted before 'mu' was registered on "
                "this branch have to be refitted, or use "
                "central_tendency='median'."
            )

        parameters = likelihood.parameters

        # A fixed sigma never reaches the posterior, but it is usable directly.
        sigma_name = f"{output_var}_sigma"
        if sigma_name in posterior:
            sigma = posterior[sigma_name]
        elif "sigma" in parameters and not isinstance(parameters["sigma"], Prior):
            sigma = parameters["sigma"]
        else:
            raise ValueError(
                f"The truncation correction needs the likelihood scale, which "
                f"is neither in the posterior as '{sigma_name}' nor a fixed "
                f"'sigma' on the prior. A tau-parameterised TruncatedNormal "
                f"lands here too. Use central_tendency='median'."
            )

        bounds = {}
        for bound, default in (("lower", -np.inf), ("upper", np.inf)):
            value = parameters.get(bound, default)
            if isinstance(value, Prior):
                raise ValueError(
                    f"The truncation correction needs a fixed '{bound}' bound, "
                    f"but it was given a prior. Use central_tendency='median'."
                )
            bounds[bound] = value

        mu = posterior["mu"]
        alpha = (bounds["lower"] - mu) / sigma
        beta = (bounds["upper"] - mu) / sigma

        # The textbook ratio cancels to zero once the truncation point is about
        # ten sigma from mu, which the identity link permits, and returns nan
        # there.  erfcx is the scaled complementary error function, which keeps
        # the one-sided cases exact and is a ufunc, so it stays vectorised.
        # scipy.stats.truncnorm is exact too but roughly 2000x slower, which
        # matters on a full posterior.
        # np.all, because a bound may be an array: a partly infinite one is not
        # one-sided everywhere, so it falls through to the two-sided branch,
        # which handles infinite entries correctly (only more slowly).
        root_two = np.sqrt(2.0)
        if np.all(np.isposinf(bounds["upper"])):
            return sigma * np.sqrt(2 / np.pi) / erfcx(alpha / root_two)
        if np.all(np.isneginf(bounds["lower"])):
            return -sigma * np.sqrt(2 / np.pi) / erfcx(-beta / root_two)

        # Two-sided truncation. The direct form returns inf or nan once both
        # bounds sit on the same side of mu, since numerator and denominator
        # both underflow. scipy handles the whole range; it is far slower, but
        # a two-sided likelihood is uncommon and correctness comes first.
        return xr.apply_ufunc(
            lambda a, b, m, s: truncnorm.mean(a, b, loc=m, scale=s) - m,
            alpha,
            beta,
            mu,
            sigma,
        )


class LogLinkSpec(LinkSpec):
    r"""Log link: ``median(y) = exp(mu) * target_scale``.

    The likelihood is ``LogNormal(mu, sigma)``, so ``exp(mu)`` is the
    conditional **median** of the response, not its mean
    (``E[y] = exp(mu + sigma**2 / 2) * target_scale``).  All predictions and
    counterfactual contributions are computed on this median scale; use the
    ``central_tendency="mean"`` option (which applies :meth:`to_mean_scale`,
    the ``exp(sigma**2 / 2)`` factor) to obtain mean-scale quantities.
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
        """Register counterfactual ``total_media_contribution_original_scale`` and ``{output_var}_original_scale``.

        The counterfactual ``exp(mu) - exp(mu - media)`` is a median-scale
        delta for whatever variable the likelihood puts ``mu`` on.  Under a
        ``Censored(LogNormal)`` likelihood that is the latent uncensored
        variable, so the result describes unconstrained demand rather than the
        observed clipped response.  Note that
        :meth:`LogLinkSpec.validate_target` rejects any non-positive target,
        so the zero-inflated data that left-censoring at zero is meant for
        cannot be fitted under this link at all.
        """
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

    def to_mean_scale(
        self,
        dataset: xr.Dataset,
        posterior: xr.Dataset,
        likelihood: Prior,
        target_scale: xr.DataArray,
        output_var: str = "y",
    ) -> xr.Dataset:
        """Multiply by the LogNormal mean/median ratio.

        The log-link model is multiplicative in the components, so the
        proportional form is the right one here and *likelihood* is not
        consulted: :meth:`validate_likelihood_compatibility` already pins the
        log link to ``LogNormal``.
        """
        return dataset * self._mean_ratio(posterior, output_var)

    def _mean_ratio(
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
