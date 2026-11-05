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

from collections import Counter
from importlib import metadata
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar, cast

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor.xtensor as ptx
import xarray as xr
from packaging.version import Version
from pydantic import Field, InstanceOf, field_validator
from pytensor.xtensor.type import XTensorVariable
from scipy.stats import halfnorm

from pymc_marketing.mmm.additive_effect import MuEffect, safe_to_datetime
from pymc_marketing.mmm.link import LinkFunction
from pymc_marketing.serialization import (
    DeserializationContext,
    SerializationError,
    serialization,
)

if TYPE_CHECKING:
    from impulso import VAR, FittedVAR, VARData

    from pymc_marketing.mmm.mmm import MMM

# The lower bound of the impulso pins in pyproject.toml; a test keeps them in step.
_MIN_IMPULSO = "0.1.3"

# A half-normal with scale U / _HALFNORMAL_Q95 exceeds U with probability 0.05.
_HALFNORMAL_Q95 = float(halfnorm.ppf(0.95))


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


def _uncenter(
    posterior: xr.Dataset, endog_mean: np.ndarray, exog_mean: np.ndarray | None
) -> xr.Dataset:
    """Return the posterior of a VAR fitted to centered data, on the raw data.

    A VAR fitted to ``y - mu_y`` and ``x - mu_x`` with intercept ``c``, lag
    coefficients ``A_1, ..., A_p`` and exogenous coefficients ``B_exog`` is, on ``y``
    and ``x``, the VAR with the same coefficients and the intercept
    ``c + (I - A_1 - ... - A_p) mu_y - B_exog mu_x``. It is computed for every draw.

    Parameters
    ----------
    posterior : xr.Dataset
        Draws of Impulso's ``intercept``, ``B`` and, with exogenous data, ``B_exog``.
    endog_mean : np.ndarray
        ``mu_y``, in the order of the posterior's ``var`` coordinate.
    exog_mean : np.ndarray or None
        ``mu_x``, in the order of the posterior's ``exog`` coordinate, or ``None``
        without exogenous data.

    Returns
    -------
    xr.Dataset
        ``posterior`` with the intercept on the raw data.
    """
    n_lags = posterior.sizes["coeff"] // posterior.sizes["var"]
    # `B`'s columns are lag-major, so `B` times `mu_y` repeated once per lag is
    # the sum of `A_l mu_y` over the lags.
    lagged_mean = xr.DataArray(np.tile(endog_mean, n_lags), dims="coeff")
    intercept = (
        posterior["intercept"]
        + xr.DataArray(endog_mean, dims="var")
        - xr.dot(posterior["B"], lagged_mean, dim="coeff")
    )
    if exog_mean is not None:
        intercept = intercept - xr.dot(
            posterior["B_exog"], xr.DataArray(exog_mean, dims="exog"), dim="exog"
        )
    # Impulso reads the intercept's draws by position.
    return posterior.assign(intercept=intercept.transpose("chain", "draw", "var"))


def _format_dates(dates: pd.DatetimeIndex, limit: int = 5) -> str:
    """Return the first ``limit`` dates, and how many more there are, for a message.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        The dates to list.
    limit : int, default 5
        The most dates to list.

    Returns
    -------
    str
        The dates as ``YYYY-MM-DD``, separated by commas.
    """
    listed = ", ".join(dates[:limit].strftime("%Y-%m-%d"))
    if len(dates) > limit:
        listed += f" and {len(dates) - limit} more"
    return listed


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
    out of ``X``, and the effect matches their rows to the MMM's dates. A forecast of
    the baseline would need new brand data anyway. The brand data are therefore
    constants in the graph, not data in the MMM's ``constant_data``. Saving the MMM
    stores them in an idata group of their own.

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
    sales the brand caused, and it sums to about zero over time. The brand's return
    comes from ``fitted_var(mmm).dynamic_multiplier(...)`` instead, see
    :meth:`fitted_var`.

    The baseline's innovation standard deviation, ``{prefix}::sigma_sd_0``, has a
    half-normal prior with scale U / 1.96, so it exceeds U with prior probability
    0.05. U is ``baseline_innovation_sd`` or, by default, the residual standard
    deviation of an AR(1) fit to the target. U sets this prior, but not the
    baseline's scale in the priors on the VAR's lag and exogenous coefficients: that
    is the target's AR(1) residual standard deviation, whatever
    ``baseline_innovation_sd`` is. ``baseline_innovation_sd`` stands in for it only
    when the target does not vary, as for the all-zero target the MMM is built on
    without ``y``.

    The first ``lags`` values of the baseline have a normal prior with mean 0 whose
    standard deviation is the target's, in the target's original units: a
    data-derived scale on the order of the baseline's marginal standard deviation,
    since the baseline is a component of the target. The start of the path trades
    off against the MMM's intercept, so it gets the scale of the rest of the path
    rather than a looser one. When the target does not vary, as for the all-zero
    target the MMM builds on without ``y``, the standard deviation is
    ``U / sqrt(1 - baseline_own_lag_mean**2)`` instead. That is the baseline's
    stationary standard deviation if it were an AR(1) with own-lag coefficient
    ``baseline_own_lag_mean`` and innovation standard deviation U.

    Parameters
    ----------
    brand_data : pd.DataFrame
        The brand metrics and exogenous inputs, with the MMM's date column. Rows are
        matched to the MMM's dates on that column, so they can be in any order and
        cover other dates too. Construction checks only that the columns named in
        ``endog_names`` and ``exog_names`` exist and are numeric. Building the MMM
        checks that, on the MMM's dates, those columns are finite and vary, and that
        there is exactly one row per MMM date. Rows on other dates are neither used
        nor checked, so they may hold NaN or repeat a date. The dates must match the
        MMM's in time-zone awareness: both naive or both aware. The index is not
        used. The effect keeps a copy of the frame.
    baseline_name : str
        Name of the latent baseline in the VAR. It must not be a column of
        ``brand_data``.
    endog_names : list[str]
        The VAR's endogenous series in order: ``baseline_name`` first, then at least
        one column of ``brand_data``.
    exog_names : list[str]
        Columns of ``brand_data`` that enter the VAR as exogenous regressors. May be
        empty. No name may appear twice across ``endog_names`` and ``exog_names``.
    prefix : str
        Prefix for the effect's variables. The VAR's variables are named
        ``{prefix}::B``, ``{prefix}::latent`` and so on, and the baseline's
        contribution is ``{prefix}_effect_contribution``.
    var : impulso.VAR
        The VAR specification, with an integer ``lags``, constant volatility and
        Gaussian errors. It is not modified. Its volatility and error distribution
        must be Impulso's own, since the fitted MMM stores the spec and no other
        volatility or error distribution loads back as it was saved. For now, its
        prior must be a ``MinnesotaPrior`` or the ``"minnesota"`` shorthand, since
        Impulso does not yet load other priors back as they were saved, see
        https://github.com/QuantClimate/Impulso/issues/379.
    baseline_own_lag_mean : float, default 0.0
        Prior mean of the baseline's own first-lag coefficient, strictly between -1
        and 1 because the baseline is a stationary deviation. The effect sets the
        baseline's entry of the Minnesota prior's ``own_lag_mean`` to it. A scalar
        ``own_lag_mean`` applies to every brand metric. A per-series
        ``own_lag_mean`` must start with this value, or construction raises. When the
        target does not vary, it also sets the standard deviation of the start of the
        baseline's path.
    baseline_innovation_sd : float, optional
        U, the bound on the baseline's innovation standard deviation, in the target's
        original units. It must be positive and finite. By default, U is the residual
        standard deviation of an AR(1) fit to the target. It sets the baseline's
        innovation prior, not the priors on the VAR's coefficients. It sets the
        standard deviation of the start of the baseline's path only when the target
        does not vary, as for the all-zero target without ``y``. The effect owns the
        first entry of the ``innovation_scale_priors`` of ``var``'s ``Constant``
        volatility, which have one entry per ``endog_names`` entry, the baseline's
        first. If ``var`` sets them, that entry must be Impulso's default,
        ``InnovationScalePrior(family="halfcauchy", scale=sigma_sd_beta)``, which the
        effect replaces, or construction raises. The brand metrics keep their
        entries, or Impulso's default when there are none.

    Notes
    -----
    The baseline exists only on the dates the MMM is fitted on. On those dates,
    fitting, the contribution decomposition, the MMM's summaries, :meth:`fitted_var`
    and posterior predictive sampling work. Saving and loading the MMM work, and so
    does posterior predictive sampling after loading. Channel incrementality works on
    any window and frequency: the baseline does not depend on spend, so it adds
    nothing to a channel's increment. The budget optimizer runs on any window with
    the default objective, which ignores the baseline, as it should: brand spend is
    an exogenous input of the VAR, not a channel. An objective that includes the
    baseline, such as ``response_variable="total_response_original_scale"``, raises
    a ``NotImplementedError``. So do posterior predictive sampling and ``predict`` on
    any other set of dates, including a subset of the fitted ones or with
    ``include_last_observations=True``, and time-slice cross-validation, which
    predicts on each fold's test dates.

    The MMM must have no ``dims``, ``time_varying_intercept=False`` and
    ``link="identity"``, and building any other MMM raises. The VAR has no
    cross-sectional dimension, so it models a single panel. A time-varying intercept
    would compete with the baseline, which is a deviation around a constant
    intercept. The baseline enters the MMM's mean additively in the target's units,
    which holds only on the identity link.

    U is an upper bound on the baseline's innovation standard deviation rather than
    an estimate of it, because the target's AR(1) residual standard deviation
    includes the sales noise as well as the baseline's innovations. The data separate
    the two only weakly, so the prior partly decides how much of the target's
    movement goes to the baseline and how much to the likelihood's noise. Check how
    sensitive the results are to it: refit with a different
    ``baseline_innovation_sd``, such as half the default, and compare
    ``{prefix}::sigma_sd_0``, the likelihood's noise and the baseline equation's
    coefficients. Halving it changes only the baseline's innovation prior. The priors
    on the VAR's coefficients, such as the baseline's loadings on the brand metrics,
    take their scale from the target, and so does the start of the baseline's path
    unless the target does not vary.

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

    An MMM takes at most one ``VARBaselineEffect``, and building an MMM with two
    raises. Two baselines in one mean would be identified only through their sum,
    and two VARs over the same brand series would count those series' likelihood
    twice, so put every brand series in one VAR.

    Impulso adds a positional ``time`` coordinate that no variable uses.

    Saving the MMM stores ``brand_data``, with its column dtypes but not its index,
    in the idata group ``supplementary_data_{prefix}_brand_data``. netCDF has no
    missing text value, so missing values in a text column load as empty strings.
    Columns that netCDF cannot store, such as time-zone-aware dates and
    categoricals, make saving fail, so convert or drop them first.

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

        # The VAR in the data's units, as an Impulso FittedVAR whose baseline is a
        # deviation around the MMM's intercept: the response of the baseline and
        # the brand metrics to a unit of brand spend
        brand_var.fitted_var(mmm).dynamic_multiplier(horizon=26).median()
    """

    brand_data: InstanceOf[pd.DataFrame]
    baseline_name: str
    endog_names: list[str]
    exog_names: list[str]
    prefix: str
    # Impulso is optional, so the spec's type is checked in `model_post_init`.
    var: Any
    baseline_own_lag_mean: float = Field(0.0, gt=-1, lt=1)
    baseline_innovation_sd: float | None = Field(None, gt=0, allow_inf_nan=False)
    supports_new_dates: ClassVar[bool] = False

    @field_validator("brand_data")
    @classmethod
    def _copy_brand_data(cls, brand_data: pd.DataFrame) -> pd.DataFrame:
        """Copy ``brand_data``.

        Without the copy, later edits to the caller's frame would bypass the checks
        and change the centering means of :meth:`fitted_var`.
        """
        return brand_data.copy()

    def model_post_init(self, context: Any, /) -> None:
        """Check ``var``, the names, and the columns of ``brand_data`` they name.

        Only the columns' presence and dtypes are checked here. Their values matter
        only on the MMM's dates, so :meth:`create_data` checks them there.

        Raises
        ------
        TypeError
            If ``var`` is not an Impulso ``VAR`` specification, if its prior is not a
            ``MinnesotaPrior``, or if it does not load back as it was saved.
        ValueError
            If ``endog_names`` does not start with ``baseline_name`` or has nothing
            after it, if ``baseline_name`` is a column of ``brand_data``, if a name
            appears more than once across ``endog_names`` and ``exog_names``, if a
            column named in them is missing from ``brand_data`` or is not numeric, if
            ``var``'s prior has a per-series ``own_lag_mean`` whose first entry is not
            ``baseline_own_lag_mean``, or if its volatility has
            ``innovation_scale_priors`` whose first entry is not Impulso's default.
        """
        self._check_var()
        self._check_names()
        self._check_brand_columns()
        self._check_baseline_entries()

    def _check_var(self) -> None:
        """Check that ``var`` is an Impulso ``VAR`` that the fitted MMM can store.

        Raises
        ------
        TypeError
            If ``var`` is not an Impulso ``VAR`` specification, if its prior is not a
            ``MinnesotaPrior``, or if it does not load back as it was saved.
        """
        impulso = _import_impulso()
        if not isinstance(self.var, impulso.VAR):
            raise TypeError(
                "var must be an impulso.VAR specification, "
                f"got {type(self.var).__name__}."
            )
        prior = self.var.resolved_prior
        if not isinstance(prior, impulso.MinnesotaPrior):
            raise TypeError(
                "var's prior must be a MinnesotaPrior or 'minnesota', got "
                f"{type(prior).__name__}: only a MinnesotaPrior is supported for now, "
                "since the fitted MMM stores var and Impulso does not yet load other "
                "priors back as they were saved, see "
                "https://github.com/QuantClimate/Impulso/issues/379."
            )
        # Fitting serializes the effect after sampling and loading validates it back,
        # so check both before the fit.
        try:
            loaded = impulso.VAR.model_validate(self.var.model_dump(mode="json"))
        except ValueError as exc:
            raise TypeError(
                "var must survive saving and loading, since the fitted MMM stores it: "
                f"{exc}."
            ) from exc
        if loaded != self.var:
            raise TypeError(
                "var must survive saving and loading, since the fitted MMM stores it, "
                "but it loads back as a different spec: a volatility or error "
                "distribution from outside Impulso does not survive."
            )

    def _check_names(self) -> None:
        """Check ``baseline_name``, ``endog_names`` and ``exog_names``.

        Raises
        ------
        ValueError
            If ``endog_names`` does not start with ``baseline_name`` or has nothing
            after it, if ``baseline_name`` is a column of ``brand_data``, or if a name
            appears more than once across ``endog_names`` and ``exog_names``.
        """
        if self.endog_names[:1] != [self.baseline_name]:
            raise ValueError(
                f"endog_names must start with baseline_name {self.baseline_name!r}, "
                f"got {self.endog_names}."
            )
        if self.baseline_name in self.brand_data.columns:
            raise ValueError(
                f"baseline_name {self.baseline_name!r} must not be a column of "
                "brand_data: the baseline is latent."
            )
        if not self._observed_names:
            raise ValueError(
                "endog_names must name at least one column of brand_data after "
                f"baseline_name {self.baseline_name!r}."
            )
        name_counts = Counter([*self.endog_names, *self.exog_names])
        if repeated := [name for name, count in name_counts.items() if count > 1]:
            raise ValueError(
                f"Names {repeated} appear more than once in endog_names and exog_names."
            )

    def _check_brand_columns(self) -> None:
        """Check that the columns the VAR uses are in ``brand_data`` and numeric.

        Raises
        ------
        ValueError
            If a column named in ``endog_names`` or ``exog_names`` is missing from
            ``brand_data`` or is not numeric.
        """
        named_columns = {
            "endog_names": self._observed_names,
            "exog_names": self.exog_names,
        }
        for field, names in named_columns.items():
            if missing := [n for n in names if n not in self.brand_data.columns]:
                raise ValueError(
                    f"Columns {missing} of {field} are missing in brand_data."
                )

        if non_numeric := [
            name
            for name, dtype in self.brand_data[self._column_names].dtypes.items()
            if not pd.api.types.is_numeric_dtype(dtype)
        ]:
            raise ValueError(f"Columns {non_numeric} of brand_data are not numeric.")

    def __eq__(self, other: object) -> bool:
        """Compare with another effect, comparing ``brand_data`` with pandas.

        ``brand_data`` is equal when it has the same columns, dtypes and rows, in the
        same order. Its index is not compared, since the effect does not use it.

        Parameters
        ----------
        other : object
            The object to compare with.

        Returns
        -------
        bool
            Whether ``other`` is a ``VARBaselineEffect`` with equal fields.
        """
        if not isinstance(other, VARBaselineEffect):
            return NotImplemented
        return self.model_dump(exclude={"brand_data"}) == other.model_dump(
            exclude={"brand_data"}
        ) and self.brand_data.reset_index(drop=True).equals(
            other.brand_data.reset_index(drop=True)
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict, without ``brand_data``.

        ``brand_data`` is stored in the idata group named by ``brand_data_group``, see
        :meth:`idata_groups`. Its column dtypes are stored here, since not all of them
        survive netCDF: dates load in nanoseconds and object columns as strings.
        ``format_version`` numbers the dict's layout, so that loading can tell a
        layout it does not read.
        """
        return {
            **self.model_dump(mode="json", exclude={"brand_data"}),
            "format_version": _FORMAT_VERSION,
            "brand_data_group": self._brand_data_group,
            "brand_data_dtypes": self.brand_data.dtypes.astype(str).to_dict(),
        }

    def idata_groups(self) -> dict[str, xr.Dataset]:
        """Return ``brand_data``, without its index, as a supplementary idata group."""
        return {
            self._brand_data_group: xr.Dataset.from_dataframe(
                self.brand_data.reset_index(drop=True)
            ),
        }

    @property
    def _brand_data_group(self) -> str:
        """The idata group of ``brand_data``.

        It is named after the data as well as the prefix, so it differs from the
        ``supplementary_data_{prefix}`` group of an ``EventAdditiveEffect``.
        """
        return f"supplementary_data_{self.prefix}_brand_data"

    def create_data(self, mmm: "MMM") -> None:  # type: ignore[override]
        """Check the MMM, and ``brand_data`` on the MMM's dates.

        The brand data enter the graph as constants in :meth:`create_effect`, so no
        data variable is registered.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.

        Raises
        ------
        ValueError
            If the MMM has more than one ``VARBaselineEffect``, ``dims``, a
            time-varying intercept or a link other than the identity, if
            ``brand_data`` cannot be matched to the MMM's dates, or if a column named
            in ``endog_names`` or ``exog_names`` has a NaN or infinite value on them
            or is constant on them.
        """
        effects = [
            effect for effect in mmm.mu_effects if isinstance(effect, VARBaselineEffect)
        ]
        if len(effects) > 1:
            others = [effect.prefix for effect in effects if effect is not self]
            found = (
                f"found others with prefixes {others}. Put every brand series in one "
                "VAR."
                if others
                else "but this one was added more than once. Add it once."
            )
            raise ValueError(
                f"VARBaselineEffect {self.prefix!r}: an MMM takes at most one "
                f"VARBaselineEffect, {found}"
            )
        if mmm.dims:
            raise ValueError(
                f"VARBaselineEffect {self.prefix!r} needs an MMM without dims, got "
                f"dims={mmm.dims}: the VAR has no cross-sectional dimension, so it "
                "models a single panel."
            )
        if mmm.time_varying_intercept:
            raise ValueError(
                f"VARBaselineEffect {self.prefix!r} needs an MMM with "
                "time_varying_intercept=False: the baseline is a zero-mean deviation "
                "around a constant intercept, and a time-varying intercept would "
                "compete with it for the same movement in the target."
            )
        if mmm.link != LinkFunction.IDENTITY:
            raise ValueError(
                f"VARBaselineEffect {self.prefix!r} needs an MMM with link='identity', "
                f"got link={mmm.link.value!r}: the baseline enters the MMM's mean "
                "additively in the target's units, which holds only on the identity "
                "link."
            )

        columns, _ = self._brand_columns(mmm)
        finite = np.isfinite(columns.to_numpy()).all(axis=0)
        if non_finite := columns.columns[~finite].tolist():
            raise ValueError(
                f"VARBaselineEffect {self.prefix!r} needs finite brand data: columns "
                f"{non_finite} of brand_data contain NaN or infinite values on the "
                "MMM's dates."
            )
        if constant_columns := columns.columns[columns.nunique() == 1].tolist():
            raise ValueError(
                f"VARBaselineEffect {self.prefix!r} needs brand data that vary over "
                f"the MMM's dates: columns {constant_columns} of brand_data are "
                "constant on them."
            )

    def create_effect(self, mmm: "MMM") -> XTensorVariable:  # type: ignore[override]
        """Build the VAR and return the baseline's contribution.

        The VAR is registered in a model named ``prefix`` nested in the MMM's model,
        which prefixes its variables.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.

        Returns
        -------
        XTensorVariable
            The baseline in scaled-target units, with dims ``("date",)``.

        Raises
        ------
        ValueError
            If the target scale is not positive, or if the target's AR(1) residual
            standard deviation is not positive and ``baseline_innovation_sd`` is not
            set, as for the all-zero target the MMM builds on when no ``y`` is
            given.
        """
        impulso = _import_impulso()
        columns, means = self._brand_columns(mmm)
        centered = columns - means
        exog = centered[self.exog_names].to_numpy() if self.exog_names else None

        target_scale = mmm.model["target_scale"]
        target_scale_value = float(target_scale.get_value())
        if not target_scale_value > 0:
            raise ValueError(
                f"VARBaselineEffect {self.prefix!r} needs a positive target scale: the "
                "baseline enters the MMM's mean divided by it, and it is "
                f"{target_scale_value}. The MMM builds on an all-zero target, whose "
                "data-derived scale is 0, when no y is given, as in "
                "sample_prior_predictive; pass y, or fix the target scale with "
                "FixedScaling and set baseline_innovation_sd."
            )

        baseline_scale = self._baseline_scale(mmm)
        latent_init_sigma = self._baseline_start_sd(mmm, baseline_scale)
        spec = self._build_spec(baseline_scale)
        with pm.Model(name=self.prefix):
            handles = spec.build_in_model(
                endog=centered[self._observed_names].to_numpy(),
                exog=exog,
                n_lags=spec.lags,
                endog_names=self.endog_names,
                exog_names=self.exog_names or None,
                endog_scales=[
                    self._baseline_endog_scale(mmm),
                    *impulso.ar1_residual_sd(columns[self._observed_names].to_numpy()),
                ],
                intercept_equations=self._observed_names,
                latent_names=[self.baseline_name],
                latent_init_sigma=latent_init_sigma,
            )

        baseline = ptx.as_xtensor(handles.latent[:, 0], dims=("date",))
        return pmd.Deterministic(
            f"{self.prefix}_effect_contribution", baseline / target_scale
        )

    def set_data(  # type: ignore[override]
        self, mmm: "MMM", model: pm.Model, X: xr.Dataset
    ) -> None:
        """Do nothing: the brand data enter the graph as constants.

        The baseline exists only on the dates the MMM was fitted on, yet this does
        not raise on other dates. The budget optimizer calls it with its own window
        and, with the default objective, evaluates only the channel contributions,
        which do not depend on the baseline. The MMM refuses prediction on any other
        set of dates instead, and objectives that include the baseline, because the
        effect sets ``supports_new_dates`` to ``False``.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.
        model : pm.Model
            The PyMC model the new data are set on.
        X : xr.Dataset
            The new data.
        """

    def frozen_deterministics(self) -> list[str]:
        """Return the baseline's contribution, which is taken from the posterior.

        The contribution is a latent path over the MMM's dates, so its posterior
        draws are used as they are, as for the MMM's HSGP terms. Recomputing it would
        fail in incrementality, which swaps in posterior draws only for ``pymc.dims``
        variables, and Impulso's variables are not.

        Returns
        -------
        list[str]
            The name of the baseline's contribution.
        """
        return [self.contribution_var_name]

    def fitted_var(self, mmm: "MMM") -> "FittedVAR":
        """Return the fitted VAR as an Impulso ``FittedVAR`` in the data's units.

        The effect fits the VAR to the brand columns centered on their means over the
        MMM's dates. This method undoes the centering for every posterior draw, so the
        result reads as a VAR fitted to the baseline and the raw brand columns. With
        ``c`` the fitted intercept, ``A_l`` the lag-``l`` coefficients, ``B_exog``
        the exogenous coefficients, and ``mu_y`` and ``mu_x`` the means of the
        endogenous and exogenous columns, the intercept on the raw data is
        ``c + (I - A_1 - ... - A_p) mu_y - B_exog mu_x``. The baseline is not
        centered and its equation has no intercept, so it takes 0 in ``mu_y`` and in
        ``c``: its intercept here follows from the coefficients rather than being
        estimated. The coefficients and the innovation covariance are unchanged.

        The means are recomputed from ``brand_data``, so it must hold the data the
        MMM was fitted with.

        Parameters
        ----------
        mmm : MMM
            The fitted MMM that carries this effect.

        Returns
        -------
        impulso.FittedVAR
            The VAR over ``endog_names``, with ``exog_names`` as its exogenous
            inputs. Its posterior holds the VAR's coefficients, intercept and
            innovation covariance under Impulso's names, without the prefix, and it
            carries the MMM's sampler statistics. Its data are the brand columns on
            the MMM's dates and, for the baseline, the posterior mean of its path in
            the target's units, as a zero-mean deviation around the MMM's intercept:
            add ``intercept_contribution_original_scale`` for the level.

        Raises
        ------
        RuntimeError
            If the MMM has not been fitted.
        ValueError
            If the MMM's posterior has no VAR named ``prefix``, or one over other
            series than ``endog_names`` and ``exog_names``, if ``brand_data`` no
            longer covers the MMM's dates, if U comes from the target and is not
            positive, or if Impulso rejects the posterior or the data.

        Notes
        -----
        Each posterior draw has its own baseline path, but a ``FittedVAR`` holds one
        data set, so its data hold the paths' posterior mean. The draws themselves
        are ``{prefix}::latent`` in the MMM's posterior. The path is a deviation
        around the MMM's intercept, so a ``forecast`` or ``historical_decomposition``
        of the baseline is one too, not a level of the target.

        Methods that read only the posterior are exact: ``dynamic_multiplier``,
        ``sigma`` and, since the effect's volatility is constant, the
        ``impulse_response`` and ``fevd`` of the ``IdentifiedVAR`` that
        ``set_identification_strategy`` returns, with any identification scheme but
        ``ProxySVAR``, which identifies from the residuals.

        Methods that read ``data.endog`` are plug-in approximations: they use the
        posterior mean path in every draw, so their uncertainty leaves out the
        baseline's, and a quantity that depends on the path non-linearly, such as
        the standardized Granger scale, can also be biased. They are ``forecast``,
        ``conditional_forecast``, ``posterior_predictive``, ``granger_causality``
        with ``standardize=True``, and the ``IdentifiedVAR``'s
        ``historical_decomposition``, ``counterfactual`` and ``structural_scenario``.

        Examples
        --------
        With ``brand_var`` and the fitted ``mmm`` of the class example, the response
        of the baseline and the brand metrics to a unit of brand spend over the
        following 26 weeks:

        .. code-block:: python

            fitted = brand_var.fitted_var(mmm)
            fitted.dynamic_multiplier(horizon=26).median()
        """
        posterior = self._effect_posterior(mmm)
        impulso = _import_impulso()
        columns, means = self._brand_columns(mmm)
        posterior = _uncenter(
            posterior,
            np.concatenate([[0.0], means[self._observed_names]]),
            means[self.exog_names].to_numpy() if self.exog_names else None,
        )

        groups = {"/posterior": posterior}
        idata = cast(xr.DataTree, mmm.idata)
        if "sample_stats" in idata:
            groups["/sample_stats"] = idata["sample_stats"].to_dataset()
        data = self._fitted_var_data(mmm, columns)
        spec = self._build_spec(self._baseline_scale(mmm))
        return impulso.FittedVAR.from_posterior(
            xr.DataTree.from_dict(groups),
            data,
            spec.lags,
            volatility=spec.resolved_volatility,
            error_dist=spec.resolved_error_dist,
        )

    def _effect_posterior(self, mmm: "MMM") -> xr.Dataset:
        """Return the posterior of the VAR fitted to the centered data.

        The variables keep Impulso's names, without the prefix, and the baseline's
        path is left out. The intercept takes 0 for the baseline, whose equation has
        none, so it covers every endogenous series as Impulso's does.

        Parameters
        ----------
        mmm : MMM
            The fitted MMM that carries this effect.

        Returns
        -------
        xr.Dataset
            The draws of the VAR's coefficients, intercept and innovation scales.

        Raises
        ------
        RuntimeError
            If the MMM has not been fitted.
        ValueError
            If the MMM's posterior has no VAR named ``prefix``, or one over other
            series than ``endog_names`` and ``exog_names``.
        """
        if mmm.idata is None or "posterior" not in mmm.idata:
            raise RuntimeError(
                f"VARBaselineEffect {self.prefix!r} reads its VAR from the MMM's "
                "posterior, but the model hasn't been fit yet, call .fit() first."
            )

        var_prefix = f"{self.prefix}::"
        posterior = mmm.idata["posterior"].to_dataset()
        if f"{var_prefix}B" not in posterior:
            raise ValueError(
                f"VARBaselineEffect {self.prefix!r} found no '{var_prefix}' variables "
                "in the MMM's posterior: was the MMM fitted with this effect?"
            )
        fitted_endog = posterior[f"{var_prefix}B"].coords["var"].values.tolist()
        fitted_exog = []
        if f"{var_prefix}B_exog" in posterior:
            fitted_exog = (
                posterior[f"{var_prefix}B_exog"].coords["exog"].values.tolist()
            )
        if (fitted_endog, fitted_exog) != (self.endog_names, self.exog_names):
            raise ValueError(
                f"VARBaselineEffect {self.prefix!r} has endog_names "
                f"{self.endog_names} and exog_names {self.exog_names}, but the MMM "
                f"was fitted with {fitted_endog} and {fitted_exog}: fit the MMM again "
                "after changing the effect."
            )

        # The baseline's path is data to Impulso, see `_fitted_var_data`.
        path_names = [
            f"{var_prefix}{name}"
            for name in ("latent", "latent_init", "latent_innovations")
        ]
        posterior = posterior[
            [
                name
                for name in map(str, posterior.data_vars)
                if name.startswith(var_prefix) and name not in path_names
            ]
        ]
        posterior = posterior.rename(
            {
                name: name.removeprefix(var_prefix)
                for name in map(str, [*posterior.data_vars, *posterior.dims])
                if name.startswith(var_prefix)
            }
        )
        # Only the brand metrics' equations have an intercept in the graph.
        intercept = (
            posterior["intercept"]
            .rename(var_intercept="var")
            .reindex(var=self.endog_names, fill_value=0.0)
        )
        return posterior.drop_dims("var_intercept").assign(intercept=intercept)

    def _fitted_var_data(self, mmm: "MMM", columns: pd.DataFrame) -> "VARData":
        """Return the data of the fitted VAR, on the MMM's dates.

        A ``VARData`` holds one data set, so the baseline takes the posterior mean of
        its path, in the target's units.

        Parameters
        ----------
        mmm : MMM
            The fitted MMM that carries this effect.
        columns : pd.DataFrame
            The brand columns on the MMM's dates, as :meth:`_brand_columns` returns
            them.

        Returns
        -------
        impulso.VARData
            The baseline and the raw brand metrics as the endogenous series, and the
            raw exogenous columns.
        """
        impulso = _import_impulso()
        latent = mmm.fit_result[f"{self.prefix}::latent"]
        baseline = latent.mean(("chain", "draw")).to_numpy()
        observed = columns[self._observed_names].to_numpy()
        return impulso.VARData(
            endog=np.column_stack([baseline, observed]),
            endog_names=self.endog_names,
            exog=columns[self.exog_names].to_numpy() if self.exog_names else None,
            exog_names=self.exog_names or None,
            index=safe_to_datetime(mmm.model.coords["date"], "date"),
        )

    @property
    def _observed_names(self) -> list[str]:
        """The endogenous series observed in ``brand_data``: all but the baseline."""
        return self.endog_names[1:]

    @property
    def _column_names(self) -> list[str]:
        """The columns of ``brand_data`` the VAR uses, observed series first."""
        return [*self._observed_names, *self.exog_names]

    def _brand_data_on_mmm_dates(self, mmm: "MMM") -> pd.DataFrame:
        """Return the rows of ``brand_data`` on the MMM's dates, in the MMM's order.

        Rows on other dates are not used, so they may repeat a date.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.

        Returns
        -------
        pd.DataFrame
            One row of ``brand_data`` per MMM date.

        Raises
        ------
        ValueError
            If ``brand_data`` has no column named like the MMM's date column, if
            exactly one of its dates and the MMM's has a time zone, or if it misses
            an MMM date or has one more than once.
        """
        date_column = mmm.date_column
        if date_column not in self.brand_data.columns:
            raise ValueError(
                f"VARBaselineEffect {self.prefix!r} matches brand_data to the MMM on "
                f"its date column {date_column!r}, which is missing in brand_data."
            )

        brand_dates = safe_to_datetime(self.brand_data[date_column], date_column)
        mmm_dates = safe_to_datetime(mmm.model.coords["date"], "date")
        if (brand_dates.tz is None) != (mmm_dates.tz is None):
            brand_kind, mmm_kind = (
                ("naive", "time-zone aware")
                if brand_dates.tz is None
                else ("time-zone aware", "naive")
            )
            raise ValueError(
                f"VARBaselineEffect {self.prefix!r} cannot match brand_data to the MMM "
                f"on {date_column!r}: brand_data's dates are {brand_kind} and the "
                f"MMM's are {mmm_kind}. Convert one of them, e.g. drop the time zone "
                "with .dt.tz_localize(None)."
            )

        # Rows on other dates may repeat a date, which `get_indexer` refuses, so they
        # are dropped first.
        on_mmm_dates = brand_dates.isin(mmm_dates)
        brand_dates = brand_dates[on_mmm_dates]
        if brand_dates.has_duplicates:
            repeated = brand_dates[brand_dates.duplicated()].unique()
            raise ValueError(
                f"VARBaselineEffect {self.prefix!r} needs one row of brand_data per "
                f"MMM date, but dates {_format_dates(repeated)} appear more than once."
            )

        rows = brand_dates.get_indexer(mmm_dates)
        if (rows == -1).any():
            missing = mmm_dates[rows == -1]
            message = (
                f"VARBaselineEffect {self.prefix!r} has no brand_data for MMM dates "
                f"{_format_dates(missing)}."
            )
            if len(missing) == len(mmm_dates):
                message += (
                    " None of the MMM's dates is in brand_data: check that brand_data "
                    "uses the same frequency and weekly anchor as the MMM, since "
                    "W-SUN dates, for example, never match W-MON ones."
                )
            raise ValueError(message)
        return self.brand_data[on_mmm_dates].iloc[rows]

    def _brand_columns(self, mmm: "MMM") -> tuple[pd.DataFrame, pd.Series]:
        """Return the VAR's brand columns on the MMM's dates and their means.

        The VAR is fitted to the columns centered on these means.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.

        Returns
        -------
        columns : pd.DataFrame
            The observed endogenous columns, then the exogenous ones, one row per
            MMM date.
        means : pd.Series
            The mean of each column over the MMM's dates.
        """
        brand_data = self._brand_data_on_mmm_dates(mmm)
        columns = brand_data[self._column_names].astype(float)
        return columns, columns.mean()

    def _baseline_scale(self, mmm: "MMM") -> float:
        """Return U, the bound on the baseline's innovation standard deviation.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.

        Returns
        -------
        float
            ``baseline_innovation_sd``, or else the AR(1) residual standard deviation
            of the target, in the target's original units.

        Raises
        ------
        ValueError
            If U comes from the target and is not positive, as for the all-zero
            target the MMM builds on when no ``y`` is given.
        """
        if self.baseline_innovation_sd is not None:
            return self.baseline_innovation_sd
        return self._baseline_endog_scale(mmm)

    def _baseline_start_sd(self, mmm: "MMM", baseline_scale: float) -> float:
        """Return the standard deviation of the baseline's first ``lags`` values.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.
        baseline_scale : float
            U, the bound on the baseline's innovation standard deviation, in the
            target's original units.

        Returns
        -------
        float
            The target's standard deviation, in its original units. If that is not
            positive and finite, as for the all-zero target the MMM builds on when no
            ``y`` is given, ``U / sqrt(1 - baseline_own_lag_mean**2)`` instead.
        """
        target_sd = float(np.std(mmm.xarray_dataset["_target"].to_numpy()))
        # Data-derived and independent of U, the target's sd is on the order of the
        # baseline's marginal sd when the baseline is a component of the target. A
        # scale from U would leave out what the brand metrics pass into the baseline.
        if np.isfinite(target_sd) and target_sd > 0:
            return target_sd
        return baseline_scale / np.sqrt(1 - self.baseline_own_lag_mean**2)

    def _baseline_endog_scale(self, mmm: "MMM") -> float:
        """Return the baseline's scale in the priors on the VAR's coefficients.

        It is the baseline's entry in ``endog_scales``, which sets the Minnesota
        prior's cross-lag scaling and the exogenous prior's scale. It does not
        depend on ``baseline_innovation_sd``, so that changing U moves only the
        innovation prior.

        Parameters
        ----------
        mmm : MMM
            The MMM model instance.

        Returns
        -------
        float
            The AR(1) residual standard deviation of the target, in the target's
            original units, or ``baseline_innovation_sd`` if that is not positive,
            as for the all-zero target the MMM builds on when no ``y`` is given.

        Raises
        ------
        ValueError
            If the target's AR(1) residual standard deviation is not positive and
            ``baseline_innovation_sd`` is not set.
        """
        impulso = _import_impulso()
        target = mmm.xarray_dataset["_target"].to_numpy()
        (residual_sd,) = impulso.ar1_residual_sd(target[:, None])
        if residual_sd > 0:
            return float(residual_sd)
        if self.baseline_innovation_sd is not None:
            return self.baseline_innovation_sd
        raise ValueError(
            f"VARBaselineEffect {self.prefix!r} needs a target that varies, or "
            "baseline_innovation_sd: the AR(1) residual standard deviation of the "
            "target scales the baseline in the VAR's priors, and it is "
            f"{residual_sd}. The MMM builds on an all-zero target when no y is "
            "given, as in sample_prior_predictive; pass y or set "
            "baseline_innovation_sd."
        )

    def _check_baseline_entries(self) -> None:
        """Check that ``var`` leaves the baseline's prior entries to the effect.

        The effect sets the baseline's entry, the first, of the Minnesota prior's
        ``own_lag_mean`` from ``baseline_own_lag_mean``, and of a ``Constant``
        volatility's ``innovation_scale_priors`` from U. A per-series
        ``own_lag_mean`` must therefore start with ``baseline_own_lag_mean``, and
        ``innovation_scale_priors`` with Impulso's default, which the effect
        replaces.

        Raises
        ------
        ValueError
            If ``var``'s prior has a per-series ``own_lag_mean`` whose first entry is
            not ``baseline_own_lag_mean``, or if its volatility has
            ``innovation_scale_priors`` whose first entry is not Impulso's default,
            ``InnovationScalePrior(family="halfcauchy", scale=sigma_sd_beta)``.
        """
        impulso = _import_impulso()
        own_lag_mean = self.var.resolved_prior.own_lag_mean
        baseline_mean = self.baseline_own_lag_mean
        if isinstance(own_lag_mean, tuple) and own_lag_mean[:1] != (baseline_mean,):
            raise ValueError(
                f"var's prior has own_lag_mean {own_lag_mean}, but the effect sets the "
                "baseline's entry, the first, from baseline_own_lag_mean, which is "
                f"{baseline_mean}. Set baseline_own_lag_mean to the baseline's prior "
                "mean, and give the same value as the first entry."
            )

        volatility = self.var.resolved_volatility
        if not isinstance(volatility, impulso.Constant):
            return
        scale_priors = volatility.innovation_scale_priors
        default = impulso.InnovationScalePrior(
            family="halfcauchy", scale=volatility.sigma_sd_beta
        )
        if scale_priors is not None and scale_priors[:1] != (default,):
            raise ValueError(
                f"var's volatility has innovation_scale_priors {scale_priors}, but the "
                "effect sets the baseline's entry, the first, from "
                "baseline_innovation_sd. Set baseline_innovation_sd to bound the "
                "baseline's innovation standard deviation, and make the first entry "
                f"Impulso's default, {default!r}, which the effect replaces."
            )

    def _build_spec(self, baseline_scale: float) -> "VAR":
        """Return ``var`` with the baseline's own-lag mean and innovation prior set.

        A scalar ``own_lag_mean`` of the Minnesota prior becomes one entry per
        series, with ``baseline_own_lag_mean`` as the baseline's. A per-series one
        already starts with it, as construction checks. The baseline's
        innovation-scale prior, the first entry of a ``Constant`` volatility's
        ``innovation_scale_priors``, becomes the half-normal whose 95th percentile
        is U. Any other volatility is used unchanged, and Impulso refuses it when
        building.

        Parameters
        ----------
        baseline_scale : float
            U, the bound on the baseline's innovation standard deviation.

        Returns
        -------
        impulso.VAR
            The specification the VAR is built with.
        """
        impulso = _import_impulso()
        n_observed = len(self._observed_names)
        update: dict[str, Any] = {}

        prior = self.var.resolved_prior
        if not isinstance(prior.own_lag_mean, tuple):
            observed_means = (prior.own_lag_mean,) * n_observed
            update["prior"] = prior.model_copy(
                update={"own_lag_mean": (self.baseline_own_lag_mean, *observed_means)}
            )

        volatility = self.var.resolved_volatility
        if isinstance(volatility, impulso.Constant):
            scale_priors = volatility.innovation_scale_priors
            if scale_priors is not None:
                observed_priors = scale_priors[1:]
            else:
                observed_priors = (
                    impulso.InnovationScalePrior(
                        family="halfcauchy", scale=volatility.sigma_sd_beta
                    ),
                ) * n_observed
            baseline_prior = impulso.InnovationScalePrior(
                family="halfnormal", scale=baseline_scale / _HALFNORMAL_Q95
            )
            update["volatility"] = volatility.model_copy(
                update={"innovation_scale_priors": (baseline_prior, *observed_priors)}
            )

        return self.var.model_copy(update=update)


# The layout of the dict `VARBaselineEffect.to_dict` writes. Raise it when the layout
# changes, and have the deserializer migrate the older layouts it still reads.
_FORMAT_VERSION = 1


def _deserialize_var_baseline_effect(
    data: dict[str, Any], context: DeserializationContext | None
) -> VARBaselineEffect:
    """Rebuild a ``VARBaselineEffect`` from its dict and its supplementary idata group.

    Parameters
    ----------
    data : dict[str, Any]
        The dict that :meth:`VARBaselineEffect.to_dict` returned.
    context : DeserializationContext or None
        The context holding the idata the MMM is loaded from.

    Returns
    -------
    VARBaselineEffect
        The effect, with ``brand_data`` read from the idata.

    Raises
    ------
    ImportError
        If Impulso is not installed.
    SerializationError
        If the dict has a format version other than the one this module reads, if
        there is no idata, or if it has no ``brand_data``.
    """
    version = data.get("format_version")
    if version != _FORMAT_VERSION:
        raise SerializationError(
            f"Cannot deserialize VARBaselineEffect saved in format version {version}: "
            f"this version of PyMC-Marketing reads only format version "
            f"{_FORMAT_VERSION}, so load it with the version that saved it."
        )

    impulso = _import_impulso()
    group_name = data["brand_data_group"]

    if context is None or context.idata is None:
        raise SerializationError(
            "Cannot deserialize VARBaselineEffect: no DataTree provided. The "
            f"brand_data DataFrame is stored in idata group '{group_name}' and "
            "requires a DeserializationContext with idata."
        )

    dtypes = data["brand_data_dtypes"]
    try:
        group = context.idata[group_name]
        # Read column by column: `to_dataframe` would make a column named like the
        # group's dimension the index.
        columns = {name: group[name].to_numpy() for name in dtypes}
    except KeyError as e:
        raise SerializationError(
            f"Cannot read supplementary data group '{group_name}' from DataTree: {e}"
        ) from e

    brand_data = pd.DataFrame(columns).astype(dtypes)
    var = impulso.VAR.model_validate(data["var"])
    fields = {
        name: value
        for name, value in data.items()
        if name in VARBaselineEffect.model_fields
    }
    return VARBaselineEffect(**(fields | {"brand_data": brand_data, "var": var}))


def _register_var_baseline_effect() -> None:
    serialization.register(
        f"{VARBaselineEffect.__module__}.{VARBaselineEffect.__qualname__}",
        VARBaselineEffect,
        deserializer=_deserialize_var_baseline_effect,
    )


_register_var_baseline_effect()
