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
"""Vector autoregression (VAR) baseline support, built on Impulso.

`Impulso <https://github.com/QuantClimate/Impulso>`_ is an optional dependency of
PyMC-Marketing, and the VAR baseline needs ``impulso>=0.1.3``. Install it with
``pip install 'pymc-marketing[var]'``. This module imports Impulso only when it is
used, so PyMC-Marketing imports without it.
"""

from importlib import metadata
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor.xtensor as ptx
import xarray as xr
from packaging.version import Version
from pydantic import Field, InstanceOf
from pytensor.xtensor.type import XTensorVariable

from pymc_marketing.mmm.additive_effect import Model, MuEffect

if TYPE_CHECKING:
    from impulso import VAR

# The lower bound of the impulso pins in pyproject.toml; a test keeps them in step.
_MIN_IMPULSO = "0.1.3"


def _import_impulso() -> ModuleType:
    """Import Impulso, the optional dependency of the VAR baseline.

    Impulso's version is read from its installed distribution. An Impulso without
    one, such as a source checkout put on ``sys.path``, is used unchecked, since
    its version cannot be known. A pre-release of the minimum version, such as a
    development build made before the release was tagged, is older than it and is
    refused, since it may lack what the release adds.

    Returns
    -------
    ModuleType
        The ``impulso`` module.

    Raises
    ------
    ImportError
        If Impulso is not installed, or is older than the minimum version.
    """
    try:
        import impulso
    except ImportError as exc:
        raise ImportError(
            "impulso is required for the VAR baseline. "
            "Install it with: pip install 'pymc-marketing[var]'"
        ) from exc
    try:
        installed = metadata.version("impulso")
    except metadata.PackageNotFoundError:
        return impulso
    version = Version(installed)
    if version < Version(_MIN_IMPULSO):
        prerelease = (
            f"Pre-releases of {_MIN_IMPULSO} are refused. "
            if version.is_prerelease
            and Version(version.base_version) == Version(_MIN_IMPULSO)
            else ""
        )
        raise ImportError(
            f"impulso>={_MIN_IMPULSO} is required for the VAR baseline, but "
            f"impulso {installed} is installed. {prerelease}"
            "Upgrade it with: pip install -U 'pymc-marketing[var]'"
        )
    return impulso


class VARBaselineEffect(MuEffect):
    """Latent sales baseline that moves with brand metrics through a Bayesian VAR.

    Brand marketing builds mindset metrics such as awareness and consideration, and
    those metrics move the sales baseline over the long term [1]_. This effect adds
    that baseline to the MMM's mean as a latent series and fits it jointly with the
    rest of the MMM. Its prior is a vector autoregression (VAR) over the baseline and
    the brand metrics, with inputs such as brand spend as exogenous regressors, built
    by Impulso's ``VAR.build_in_model``. The
    :ref:`brand metrics notebook <mmm_brand_metrics_long_term>` fits the same model
    in two stages: an MMM whose time-varying intercept estimates the baseline, then a
    VAR on that estimate.

    The brand data sit beside the MMM's data rather than in ``X``. Brand trackers
    often cover other periods than the MMM, or come at another cadence, so they stay
    out of ``X``. A forecast of the baseline would need new brand data anyway. The
    brand data are therefore constants in the graph, not data in the MMM's
    ``constant_data``.

    The VAR runs on the baseline in the target's original units and on the brand
    columns in their own units, centered on their means over the MMM's dates. The
    baseline equation has no intercept, so the baseline is a zero-mean deviation
    around the MMM's intercept. Its own first-lag coefficient has prior mean
    ``baseline_own_lag_mean`` rather than the Minnesota prior's random walk.

    The baseline responds to the centered brand metrics and exogenous inputs, so the
    MMM's intercept absorbs the brand's average effect. When a brand metric that
    raises sales, such as consideration, is below its mean, as between brand
    flights, it pushes the baseline below zero. The ``{prefix}_effect`` column of
    ``mmm.compute_mean_contributions_over_time()`` is therefore a deviation, not the
    sales the brand caused, and it sums to about zero over time.

    The first ``lags`` values of the baseline have a normal prior with mean 0 whose
    standard deviation is the target's, in the target's original units: a
    data-derived scale on the order of the baseline's marginal standard deviation,
    since the baseline is a component of the target. The start of the path trades
    off against the MMM's intercept, so it gets the scale of the rest of the path
    rather than a looser one.

    Parameters
    ----------
    brand_data : pd.DataFrame
        The brand metrics and exogenous inputs, one row per MMM date and in date
        order. Rows are matched to the MMM's dates by position.
    baseline_name : str
        Name of the latent baseline in the VAR. It must not be a column of
        ``brand_data``.
    endog_names : list[str]
        The VAR's endogenous series in order: ``baseline_name`` first, then columns
        of ``brand_data``.
    exog_names : list[str]
        Columns of ``brand_data`` that enter the VAR as exogenous regressors. May be
        empty.
    prefix : str
        Prefix for the effect's variables. The VAR's variables are named
        ``{prefix}::B``, ``{prefix}::latent`` and so on, and the baseline's
        contribution is ``{prefix}_effect_contribution``.
    var : impulso.VAR
        The VAR specification, with an integer ``lags``, constant volatility and
        Gaussian errors. It is not modified.
    baseline_own_lag_mean : float, default 0.0
        Prior mean of the baseline's own first-lag coefficient, strictly between -1
        and 1 because the baseline is a stationary deviation. The effect sets the
        baseline's entry of a Minnesota prior's ``own_lag_mean`` to it. A scalar
        ``own_lag_mean`` applies to every brand metric. A per-series
        ``own_lag_mean`` must start with this value, or construction raises. Any
        other prior keeps its own-lag means.

    Notes
    -----
    The baseline exists only on the dates the MMM is fitted on. On those dates,
    fitting, the contribution decomposition, the MMM's summaries and posterior
    predictive sampling work. Channel incrementality is not supported yet. The
    budget optimizer runs on any window but ignores the baseline, as it should: brand
    spend is an exogenous input of the VAR, not a channel. Posterior predictive
    sampling and ``predict`` on other dates are not supported, and they fail inside
    PyTensor with an error that does not name the effect.

    The brand metrics' likelihood and the constraint that keeps the baseline
    stationary are potentials, which prior and posterior predictive sampling ignore.
    Prior predictive baseline paths therefore come from the untruncated prior and can
    explode, and the brand metrics are never drawn. Posterior predictive sampling
    warns that it ignores them, which is harmless here: it draws from the posterior,
    which the potentials shaped.

    PyMC's NUTS sampler and nutpie work. ``nuts_sampler="numpyro"`` and
    ``nuts_sampler="blackjax"`` fail at the first gradient evaluation with JAX's
    ``NotImplementedError``: "Derivatives of non-symmetric eigenvectors are only
    valid under assumptions on the input that JAX cannot check". JAX differentiates
    the eigendecomposition in the stationarity constraint, which PyTensor's gradient
    treats as a constant. The fix belongs in Impulso, see
    https://github.com/QuantClimate/Impulso/issues/378.

    Impulso's coordinates, such as ``var``, ``coeff`` and ``exog``, are not
    prefixed, so every ``VARBaselineEffect`` in one MMM must name the same series.
    Impulso also adds a positional ``time`` coordinate that no variable uses.

    References
    ----------
    .. [1] Cain, P. M. (2022). "Modelling short- and long-term marketing effects in
       the consumer purchase journey." *International Journal of Research in
       Marketing*, 39(1), 96-116. https://doi.org/10.1016/j.ijresmar.2021.06.006

    Examples
    --------
    .. code-block:: python

        from impulso import VAR

        from pymc_marketing.mmm import (
            MMM,
            GeometricAdstock,
            LogisticSaturation,
            VARBaselineEffect,
        )

        brand_var = VARBaselineEffect(
            brand_data=df[["date", "awareness", "consideration", "brand_spend"]],
            baseline_name="baseline",
            endog_names=["baseline", "awareness", "consideration"],
            exog_names=["brand_spend"],
            prefix="brand_var",
            var=VAR(lags=1),
        )
        mmm = MMM(
            date_column="date",
            target_column="y",
            channel_columns=["x1", "x2"],
            adstock=GeometricAdstock(l_max=8),
            saturation=LogisticSaturation(),
        ).add_mu_effect(brand_var)
        mmm.fit(df[["date", "x1", "x2"]], df["y"])

        # The baseline, a deviation in the target's units, next to the other
        # components
        mmm.compute_mean_contributions_over_time()["brand_var_effect"]
    """

    brand_data: InstanceOf[pd.DataFrame]
    baseline_name: str
    endog_names: list[str]
    exog_names: list[str]
    prefix: str
    # Impulso is optional, so the spec's type is checked in `model_post_init`.
    var: Any
    baseline_own_lag_mean: float = Field(0.0, gt=-1, lt=1)

    def model_post_init(self, context: Any, /) -> None:
        """Check ``var``.

        Raises
        ------
        TypeError
            If ``var`` is not an Impulso ``VAR`` specification.
        ValueError
            If ``var``'s prior has a per-series ``own_lag_mean`` whose first entry is
            not ``baseline_own_lag_mean``.
        """
        impulso = _import_impulso()
        if not isinstance(self.var, impulso.VAR):
            raise TypeError(
                "var must be an impulso.VAR specification, "
                f"got {type(self.var).__name__}."
            )
        self._check_baseline_entries()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict, without ``brand_data``."""
        return self.model_dump(mode="json", exclude={"brand_data"})

    def create_data(self, mmm: Model) -> None:
        """Check that ``brand_data`` has one row per MMM date.

        The brand data enter the graph as constants in :meth:`create_effect`, so no
        data variable is registered.

        Parameters
        ----------
        mmm : Model
            The MMM model instance.

        Raises
        ------
        ValueError
            If ``brand_data`` and the MMM have different numbers of dates.
        """
        n_dates = len(mmm.model.coords["date"])
        if len(self.brand_data) != n_dates:
            raise ValueError(
                f"VARBaselineEffect {self.prefix!r} matches brand_data to the MMM's "
                f"dates by position, so it needs one row per date: got "
                f"{len(self.brand_data)} rows of brand_data for {n_dates} MMM dates."
            )

    def create_effect(self, mmm: Model) -> XTensorVariable:
        """Build the VAR and return the baseline's contribution.

        The VAR is registered in a model named ``prefix`` nested in the MMM's model,
        which prefixes its variables.

        Parameters
        ----------
        mmm : Model
            The MMM model instance.

        Returns
        -------
        XTensorVariable
            The baseline in scaled-target units, with dims ``("date",)``.

        Raises
        ------
        ValueError
            If the target's AR(1) residual standard deviation is not positive, as for
            the all-zero target the MMM builds on when no ``y`` is given.
        """
        impulso = _import_impulso()
        observed = self.brand_data[self._observed_names].to_numpy(dtype=float)
        exog = None
        if self.exog_names:
            exog = self.brand_data[self.exog_names].to_numpy(dtype=float)
            exog = exog - exog.mean(axis=0)

        target = mmm.xarray_dataset["_target"].to_numpy()
        (baseline_scale,) = impulso.ar1_residual_sd(target[:, None])
        if not baseline_scale > 0:
            raise ValueError(
                f"VARBaselineEffect {self.prefix!r} needs a target that varies: the "
                "AR(1) residual standard deviation of the target sets the scale of "
                f"the baseline, and it is {baseline_scale}. The MMM builds on an "
                "all-zero target when no y is given, as in sample_prior_predictive; "
                "pass y."
            )

        target_scale = mmm.model["target_scale"]
        latent_init_sigma = self._baseline_start_sd(mmm, baseline_scale)
        spec = self._build_spec()
        with pm.Model(name=self.prefix):
            handles = spec.build_in_model(
                endog=observed - observed.mean(axis=0),
                exog=exog,
                n_lags=spec.lags,
                endog_names=self.endog_names,
                exog_names=self.exog_names or None,
                endog_scales=[baseline_scale, *impulso.ar1_residual_sd(observed)],
                intercept_equations=self._observed_names,
                latent_names=[self.baseline_name],
                latent_init_sigma=latent_init_sigma,
            )

        baseline = ptx.as_xtensor(handles.latent[:, 0], dims=("date",))
        return pmd.Deterministic(
            f"{self.prefix}_effect_contribution", baseline / target_scale
        )

    def set_data(self, mmm: Model, model: pm.Model, X: xr.Dataset) -> None:
        """Do nothing: the brand data enter the graph as constants.

        The baseline exists only on the dates the MMM was fitted on, yet this does
        not raise on other dates. The budget optimizer calls it with its own window
        and evaluates only the channel contributions, which do not depend on the
        baseline.

        Parameters
        ----------
        mmm : Model
            The MMM model instance.
        model : pm.Model
            The PyMC model the new data are set on.
        X : xr.Dataset
            The new data.
        """

    @property
    def _observed_names(self) -> list[str]:
        """The endogenous series observed in ``brand_data``: all but the baseline."""
        return self.endog_names[1:]

    def _baseline_start_sd(self, mmm: Model, baseline_scale: float) -> float:
        """Return the standard deviation of the baseline's first ``lags`` values.

        Parameters
        ----------
        mmm : Model
            The MMM model instance.
        baseline_scale : float
            The residual standard deviation of an AR(1) fit to the target, in the
            target's original units.

        Returns
        -------
        float
            The target's standard deviation, in its original units. If that is not
            positive and finite, ``baseline_scale / sqrt(1 - baseline_own_lag_mean**2)``
            instead.
        """
        target_sd = float(np.std(mmm.xarray_dataset["_target"].to_numpy()))
        # Data-derived, the target's sd is on the order of the baseline's marginal sd
        # when the baseline is a component of the target. A scale from the target's
        # AR(1) residual sd would leave out what the brand metrics pass into the
        # baseline.
        if np.isfinite(target_sd) and target_sd > 0:
            return target_sd
        return baseline_scale / np.sqrt(1 - self.baseline_own_lag_mean**2)

    def _check_baseline_entries(self) -> None:
        """Check that ``var`` leaves the baseline's prior entries to the effect.

        The effect sets the baseline's entry, the first, of a Minnesota prior's
        ``own_lag_mean`` from ``baseline_own_lag_mean``. A per-series
        ``own_lag_mean`` must therefore start with ``baseline_own_lag_mean``.

        Raises
        ------
        ValueError
            If ``var``'s prior has a per-series ``own_lag_mean`` whose first entry is
            not ``baseline_own_lag_mean``.
        """
        impulso = _import_impulso()
        prior = self.var.resolved_prior
        if not isinstance(prior, impulso.MinnesotaPrior):
            return
        own_lag_mean = prior.own_lag_mean
        baseline_mean = self.baseline_own_lag_mean
        if isinstance(own_lag_mean, tuple) and own_lag_mean[:1] != (baseline_mean,):
            raise ValueError(
                f"var's prior has own_lag_mean {own_lag_mean}, but the effect sets the "
                "baseline's entry, the first, from baseline_own_lag_mean, which is "
                f"{baseline_mean}. Set baseline_own_lag_mean to the baseline's prior "
                "mean, and give the same value as the first entry."
            )

    def _build_spec(self) -> "VAR":
        """Return ``var`` with the baseline's own-lag prior mean set.

        A scalar ``own_lag_mean`` of a Minnesota prior becomes one entry per series,
        with ``baseline_own_lag_mean`` as the baseline's. A per-series one already
        starts with it, as construction checks. Only a ``MinnesotaPrior`` has an
        own-lag mean, so any other prior is used unchanged.
        """
        impulso = _import_impulso()
        prior = self.var.resolved_prior
        if not isinstance(prior, impulso.MinnesotaPrior) or isinstance(
            prior.own_lag_mean, tuple
        ):
            return self.var

        observed_means = (prior.own_lag_mean,) * len(self._observed_names)
        prior = prior.model_copy(
            update={"own_lag_mean": (self.baseline_own_lag_mean, *observed_means)}
        )
        return self.var.model_copy(update={"prior": prior})
