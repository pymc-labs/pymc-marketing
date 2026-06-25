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
"""Beta-discrete-Weibull (BdW) model for contractual customer retention.

Extension of :class:`pymc_marketing.clv.models.ShiftedBetaGeoModel` that
adds a duration-dependence shape parameter ``c`` (the discrete Weibull
exponent of Nakagawa & Osaki (1975)).  When ``c == 1`` the BdW reduces to
the sBG model.  When ``c > 1`` individual-level churn probabilities
*increase* with tenure; when ``c < 1`` they *decrease* with tenure.  The
BdW is flexible enough to describe U-shaped cohort retention curves —
situations in which cohort-level retention dips in the early periods
before increasing — which the sBG model cannot capture.

See Fader, P. S., Hardie, B. G. S., Liu, Y., Davin, J., & Steenburgh, T.
(2018).  *"How to Project Customer Retention" Revisited: The Role of
Duration Dependence.*  https://brucehardie.com/papers/037/
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pymc as pm
import xarray
from pymc.util import RandomState
from pymc_extras.prior import Prior

from pymc_marketing.clv.distributions import (
    BetaDiscreteWeibull,
    DiscreteWeibull,
)
from pymc_marketing.clv.models import CLVModel
from pymc_marketing.clv.utils import to_xarray
from pymc_marketing.model_config import ModelConfig

__all__ = [
    "BetaDiscreteWeibullModel",
    "BetaDiscreteWeibullModelIndividual",
]


# ---------------------------------------------------------------------------
# Helper utilities for the BdW closed-form expressions.  Kept as pure numpy /
# scipy so they can be used both inside and outside the PyMC model object.
# ---------------------------------------------------------------------------


def _bdw_log_survival(
    alpha: np.ndarray | xarray.DataArray,
    beta: np.ndarray | xarray.DataArray,
    c: np.ndarray | xarray.DataArray,
    t: np.ndarray | xarray.DataArray,
) -> np.ndarray | xarray.DataArray:
    """Log of the BdW survival function log S(t | alpha, beta, c).

    S(t) = B(alpha, beta + t^c) / B(alpha, beta), t = 0, 1, 2, ...

    Accepts either NumPy arrays or xarray DataArrays and preserves the
    broadcasting semantics of whichever is passed in.
    """
    from scipy.special import betaln

    t_c = t**c
    return betaln(alpha, beta + t_c) - betaln(alpha, beta)


def _bdw_survival(alpha, beta, c, t):
    """Survival function S(t) of the BdW, equation (10) of [FHLDS 2018]."""
    return np.exp(_bdw_log_survival(alpha, beta, c, t))


def _bdw_retention_rate(alpha, beta, c, t):
    """Cohort retention rate r(t) = S(t) / S(t-1), eq. (12) of [FHLDS 2018]."""
    from scipy.special import betaln

    t_c = t**c
    t_minus_1_c = np.where(t > 1, np.maximum(t - 1.0, 0.0) ** c, 0.0)
    log_ratio = betaln(alpha, beta + t_c) - betaln(alpha, beta + t_minus_1_c)
    return np.exp(log_ratio)


# ---------------------------------------------------------------------------
# Cohort-level BdW model
# ---------------------------------------------------------------------------


class BetaDiscreteWeibullModel(CLVModel):
    """Beta-discrete-Weibull (BdW) model for contract retention over discrete time.

    Cohort-level model whose individual-level dropout probabilities are
    Beta-distributed with shape parameters ``(alpha, beta)`` and whose
    contract durations follow a discrete Weibull distribution with shared
    shape parameter ``c``.  ``c == 1`` recovers the shifted-beta-geometric
    (sBG) model.  The model is described in [1]_.

    This class mirrors the interface of
    :class:`pymc_marketing.clv.models.ShiftedBetaGeoModel`:

    * it accepts the same ``customer_id`` / ``recency`` / ``T`` / ``cohort``
      columns;
    * it supports the same "unpooled per-cohort" vs. "hierarchically
      pooled" configurations via the ``(alpha, beta)`` vs. ``(phi, kappa)``
      parameterisation;
    * all predictive methods (``expected_retention_rate``,
      ``expected_probability_alive``, ``expected_num_renewals``,
      ``expected_lifetime_value``, ``expected_retention_elasticity``,
      ``distribution_new_customer_churn_time``,
      ``distribution_new_customer_theta``) have the same signatures.

    The only additional parameter in ``model_config`` is the prior on the
    duration-dependence shape parameter ``c`` (``c`` can be a scalar shared
    across cohorts or dimensioned over ``cohort``).

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the columns:

        * ``customer_id``: Unique customer identifier.
        * ``recency``: The last time period in which the customer was
          observed to be active (``1 <= recency <= T``).
        * ``T``: The maximum observed time period in the cohort.  All
          customers in a cohort share the same value of ``T``.
        * ``cohort``: Cohort label (anything hashable, e.g. date strings).

    model_config : dict, optional
        Prior specifications.  Either supply ``alpha`` / ``beta`` for
        unpooled cohort-level behaviour, or ``phi`` / ``kappa`` to pool
        cohorts hierarchically (``alpha = phi * kappa``,
        ``beta = (1 - phi) * kappa``).  The prior on the shape parameter
        ``c`` is supplied under the key ``"c"``.  Defaults:

        ======  ==========================================================
        Key     Default prior
        ======  ==========================================================
        phi     ``Prior("Uniform", lower=0, upper=1)``
        kappa   ``Prior("Pareto", alpha=1, m=1)``
        c       ``Prior("HalfNormal", sigma=1.0)``  (non-informative, >0)
        ======  ==========================================================

    sampler_config : dict, optional
        Dictionary of sampler parameters.  Defaults to ``None``.

    Examples
    --------
    .. code-block:: python

        import pandas as pd
        from pymc_extras.prior import Prior
        from pymc_marketing.clv import BetaDiscreteWeibullModel

        model = BetaDiscreteWeibullModel(
            data=pd.DataFrame(
                dict(
                    customer_id=[1, 2, 3, ...],
                    recency=[8, 1, 4, ...],
                    T=[8, 5, 5, ...],
                    cohort=["2025-02-01", "2025-04-01", "2025-04-01", ...],
                )
            ),
            model_config={
                "alpha": Prior("HalfNormal", sigma=10),
                "beta": Prior("HalfStudentT", nu=4, sigma=10),
                "c": Prior("HalfNormal", sigma=1.0),
            },
            sampler_config={"draws": 1000, "tune": 1000, "chains": 4},
        )
        model.fit(method="mcmc")
        model.fit_summary()

        # The BdW can describe U-shaped retention curves.
        retention = model.expected_retention_rate(future_t=range(1, 25))

    References
    ----------
    .. [1] Fader, P. S., Hardie, B. G. S., Liu, Y., Davin, J., &
       Steenburgh, T. (2018).  *"How to Project Customer Retention"
       Revisited: The Role of Duration Dependence.*
       https://brucehardie.com/papers/037/BdW_JIM_2018-01-10_rev.pdf
    .. [2] Fader, P. S., & Hardie, B. G. S. (2010).  Customer-Base
       Valuation in a Contractual Setting: The Perils of Ignoring
       Heterogeneity.  *Marketing Science*, 29 (1), 85-93.
    """

    _model_type = "Beta-Discrete-Weibull Model"

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        *,
        model_config: ModelConfig | None = None,
        sampler_config: dict | None = None,
    ):
        super().__init__(
            data=data,
            model_config=model_config,
            sampler_config=sampler_config,
        )

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate Beta-Discrete-Weibull-specific data requirements."""
        self._validate_cols(
            data,
            required_cols=["customer_id", "recency", "T", "cohort"],
            must_be_unique=["customer_id"],
        )

        if np.any(
            (data["recency"] < 1) | (data["recency"] > data["T"]) | (data["T"] < 2)
        ):
            raise ValueError("Model fitting requires 1 <= recency <= T, and T >= 2.")

    @property
    def cohorts(self):
        """Unique cohort values from data (first-seen ordering)."""
        if not hasattr(self, "data") or self.data is None:
            raise AttributeError(
                "cohorts not available. Call build_model(data=...) first."
            )
        return self.data["cohort"].unique()

    @property
    def cohort_idx(self):
        """Cohort indices for each customer."""
        if not hasattr(self, "data") or self.data is None:
            raise AttributeError(
                "cohort_idx not available. Call build_model(data=...) first."
            )
        return pd.Categorical(self.data["cohort"], categories=self.cohorts).codes

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    @property
    def default_model_config(self) -> ModelConfig:
        """Hierarchical cohort pooling plus a duration-dependence prior."""
        return {
            "phi": Prior("Uniform", lower=0.0, upper=1.0, dims="cohort"),
            "kappa": Prior("Pareto", alpha=1.0, m=1.0, dims="cohort"),
            "c": Prior("HalfNormal", sigma=1.0, dims="cohort"),
        }

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def build_model(self, data: pd.DataFrame | None = None) -> None:  # type: ignore[override]
        """Build the model.

        Parameters
        ----------
        data : pd.DataFrame, optional
            Input data with customer_id, recency, T, and cohort columns.
            If not provided, uses data from model initialization (deprecated).
        """
        if data is not None:
            self._validate_data(data)
            self.data = data
        elif not hasattr(self, "data") or self.data is None:
            raise ValueError(
                f"{self._model_type}.build_model() requires data parameter. "
                "Either pass data to build_model(data=...) or fit(data=...)"
            )
        else:
            self._validate_data(self.data)

        coords = {
            "customer_id": self.data["customer_id"],
            "cohort": self.cohorts,
        }
        with pm.Model(coords=coords) as self.model:
            # --- Beta mixing distribution on theta -----------------------
            if "alpha" in self.model_config and "beta" in self.model_config:
                alpha = self.model_config["alpha"].create_variable("alpha")
                beta = self.model_config["beta"].create_variable("beta")
            else:
                phi = self.model_config["phi"].create_variable("phi")
                kappa = self.model_config["kappa"].create_variable("kappa")
                alpha = pm.Deterministic("alpha", phi * kappa, dims="cohort")
                beta = pm.Deterministic("beta", (1.0 - phi) * kappa, dims="cohort")

            # --- Duration-dependence shape -------------------------------
            # Allow ``c`` to be scalar or dimensioned over cohort depending
            # on the Prior the user supplied; the Prior object handles this
            # transparently via its ``dims`` argument.
            c = self.model_config["c"].create_variable("c")

            # --- Likelihood ---------------------------------------------
            dropout = BetaDiscreteWeibull.dist(
                alpha[self.cohort_idx] if alpha.ndim >= 1 else alpha,
                beta[self.cohort_idx] if beta.ndim >= 1 else beta,
                c[self.cohort_idx] if getattr(c, "ndim", 0) >= 1 else c,
            )
            pm.Censored(
                "recency",
                dropout,
                lower=None,
                upper=self.data["T"],
                observed=self.data["recency"],
                dims=("customer_id",),
            )

    # ------------------------------------------------------------------
    # Predictive helpers
    # ------------------------------------------------------------------
    def _extract_predictive_variables(
        self,
        pred_data: pd.DataFrame,
        customer_varnames: Sequence[str] = (),
    ) -> xarray.Dataset:
        """Collect posterior parameter draws and per-customer covariates.

        Mirrors :meth:`ShiftedBetaGeoModel._extract_predictive_variables`
        and additionally extracts ``c``.
        """
        self._validate_cols(
            pred_data,
            required_cols=["customer_id", *list(customer_varnames)],
            must_be_unique=["customer_id"],
        )

        customer_id = pred_data["customer_id"].to_numpy()
        cohort = (
            pred_data["cohort"].to_numpy() if "cohort" in pred_data.columns else None
        )

        alpha = self.fit_result["alpha"]
        beta = self.fit_result["beta"]
        c = self.fit_result["c"]

        # If alpha/beta are per-cohort, re-index them per customer.
        def _per_customer(param):
            if "cohort" in param.dims and cohort is not None:
                return param.sel(cohort=xarray.DataArray(cohort, dims="customer_id"))
            return param

        alpha = _per_customer(alpha)
        beta = _per_customer(beta)
        c_per = _per_customer(c) if "cohort" in c.dims else c

        ds_vars = {"alpha": alpha, "beta": beta, "c": c_per}
        for name in customer_varnames:
            ds_vars[name] = to_xarray(customer_id, pred_data[name])

        dataset = xarray.Dataset(ds_vars)
        if cohort is not None:
            # Mirror ShiftedBetaGeoModel._extract_predictive_variables:
            # attach the cohort labels as a coordinate on ``customer_id``
            # and swap dims so downstream predictions support
            # ``.sel(cohort=...)`` exactly like the sBG model's.
            dataset = dataset.assign_coords(
                cohort=("customer_id", np.asarray(cohort))
            ).swap_dims({"customer_id": "cohort"})
        return dataset

    # ------------------------------------------------------------------
    # Predictive methods — closed-form where possible, else numerical
    # ------------------------------------------------------------------
    def expected_retention_rate(
        self,
        data: pd.DataFrame | None = None,
        *,
        future_t: int | np.ndarray | pd.Series | None = None,
    ) -> xarray.DataArray:
        r"""Compute expected per-period retention rate :math:`r(T + t)` for each customer.

        Computed from the closed-form BdW expression (equation 12 of
        Fader et al. 2018):

        .. math::

            r(t\mid\alpha,\beta,c) =
                \frac{B(\alpha,\beta + t^{c})}
                     {B(\alpha,\beta + (t-1)^{c})}.

        Parameters
        ----------
        data : ~pandas.DataFrame, optional
            Customer data.  Defaults to the fitted data restricted to
            still-active customers (``recency == T``).
        future_t : int, numpy.ndarray or pandas.Series, optional
            Number of additional time periods past ``T``.  Defaults to 0
            (retention rate *at* the current observation horizon).
        """
        from scipy.special import betaln as _betaln_np

        if data is None:
            data = self.data.query("recency == T").copy()
        if future_t is not None:
            data = data.assign(future_t=future_t)

        dataset = self._extract_predictive_variables(
            data, customer_varnames=["T", "future_t", "cohort"]
        )
        alpha = dataset["alpha"]
        beta = dataset["beta"]
        c = dataset["c"]
        T = dataset["T"]
        t = dataset["future_t"]
        tenure = T + t  # tenure at which we evaluate r(tenure)

        # BdW retention rate uses (tenure)^c and (tenure - 1)^c.
        t_c = tenure**c
        t_minus_1_c = (tenure - 1) ** c

        log_ratio = _betaln_np(alpha, beta + t_c) - _betaln_np(
            alpha, beta + t_minus_1_c
        )
        retention_rate = np.exp(log_ratio)
        return retention_rate.transpose(
            "chain", "draw", "customer_id", "cohort", missing_dims="ignore"
        )

    def expected_probability_alive(
        self,
        data: pd.DataFrame | None = None,
        *,
        future_t: int | np.ndarray | pd.Series | None = None,
    ) -> xarray.DataArray:
        r"""Probability a customer is still active at ``T + future_t``.

        Under the BdW the survival function of a customer conditional on
        surviving through the observation period ``T`` is

        .. math::

            P(\text{alive at } T + t \mid \text{alive at } T) =
                \frac{S(T + t)}{S(T)} =
                \frac{B(\alpha, \beta + (T+t)^{c})}
                     {B(\alpha, \beta + T^{c})}.
        """
        from scipy.special import betaln as _betaln_np

        if data is None:
            data = self.data.query("recency == T").copy()
        if future_t is None:
            future_t = 0
        data = data.assign(future_t=future_t)

        dataset = self._extract_predictive_variables(
            data, customer_varnames=["T", "future_t", "cohort"]
        )
        alpha = dataset["alpha"]
        beta = dataset["beta"]
        c = dataset["c"]
        T = dataset["T"]
        t = dataset["future_t"]

        log_cond_survival = _betaln_np(alpha, beta + (T + t) ** c) - _betaln_np(
            alpha, beta + T**c
        )
        return np.exp(log_cond_survival).transpose(
            "chain", "draw", "customer_id", "cohort", missing_dims="ignore"
        )

    def expected_num_renewals(
        self,
        data: pd.DataFrame | None = None,
        *,
        future_t: int | np.ndarray | pd.Series | None = None,
    ) -> xarray.DataArray:
        r"""Compute expected number of renewals in the window ``(T, T + future_t]``.

        For each draw, the expected number of successful renewals for a
        customer who is alive at ``T`` equals

        .. math::

            \mathbb{E}[N \mid \text{alive at } T] =
                \sum_{k=1}^{t} \frac{S(T + k)}{S(T)}.

        This is evaluated by direct summation — the BdW does not admit a
        closed form in general.
        """
        from scipy.special import betaln as _betaln_np

        if data is None:
            data = self.data.query("recency == T").copy()
        if future_t is None:
            raise ValueError("future_t is required for expected_num_renewals.")

        data = data.assign(future_t=future_t)
        dataset = self._extract_predictive_variables(
            data, customer_varnames=["T", "future_t", "cohort"]
        )
        alpha = dataset["alpha"]
        beta = dataset["beta"]
        c = dataset["c"]
        T = dataset["T"]
        t = dataset["future_t"]

        max_t = int(np.asarray(t).max())
        ks = xarray.DataArray(np.arange(1, max_t + 1), dims="k")

        log_cond_survival = _betaln_np(alpha, beta + (T + ks) ** c) - _betaln_np(
            alpha, beta + T**c
        )
        cond_survival = np.exp(log_cond_survival)
        # Mask future periods beyond the per-customer ``future_t``.
        cond_survival = cond_survival.where(ks <= t, 0.0)
        expected = cond_survival.sum("k")
        return expected.transpose(
            "chain", "draw", "customer_id", "cohort", missing_dims="ignore"
        )

    def expected_lifetime_value(
        self,
        discount_rate: float | np.ndarray | pd.Series = 0.0,
        data: pd.DataFrame | None = None,
        *,
        max_periods: int = 100,
    ) -> xarray.DataArray:
        r"""Compute expected discounted residual lifetime ``E(DRL)`` per customer.

        Because the BdW survival does not admit a closed form for
        :math:`\sum_{t=0}^{\infty} S(t) / (1+d)^{t}`, we follow Appendix C
        of Fader et al. (2018) and compute the residual lifetime by direct
        summation truncated at ``max_periods``:

        .. math::

            \mathbb{E}(DRL \mid \text{alive for } n \text{ periods}) =
                \sum_{t=n}^{\infty}
                \frac{S(t \mid T > n - 1)}{(1 + d)^{t - n}}.

        ``max_periods`` should be large enough that the tail contribution
        is negligible (100 periods is typically sufficient — by period 100
        the survival function is essentially zero for any realistic
        ``(alpha, beta, c)`` with non-degenerate churn).
        """
        from scipy.special import betaln as _betaln_np

        if data is None:
            data = self.data.copy()
        data = data.assign(discount_rate=discount_rate)

        dataset = self._extract_predictive_variables(
            data, customer_varnames=["T", "discount_rate", "cohort"]
        )
        alpha = dataset["alpha"]
        beta = dataset["beta"]
        c = dataset["c"]
        T = dataset["T"]
        d = dataset["discount_rate"]

        ks = xarray.DataArray(np.arange(0, max_periods + 1), dims="k")
        # Conditional survival from tenure T onwards.
        log_cond_survival = _betaln_np(alpha, beta + (T + ks) ** c) - _betaln_np(
            alpha, beta + T**c
        )
        cond_survival = np.exp(log_cond_survival)
        discount = (1.0 + d) ** (-ks)
        edrl = (cond_survival * discount).sum("k")
        return edrl.transpose(
            "chain", "draw", "customer_id", "cohort", missing_dims="ignore"
        )

    def expected_retention_elasticity(
        self,
        discount_rate: float | np.ndarray | pd.Series = 0.0,
        data: pd.DataFrame | None = None,
        *,
        max_periods: int = 100,
        eps: float = 1e-4,
    ) -> xarray.DataArray:
        r"""Elasticity of expected residual lifetime w.r.t. retention rate.

        Pfeifer & Farris (2004) define the *retention elasticity* as the
        percentage change in expected customer future value for a one
        percent increase in the per-period retention rate.  Under the sBG
        there is a closed form in ``hyp2f1`` (Fader & Hardie 2010), but
        the BdW has time-varying retention rates, so we compute the
        elasticity by finite differences on :meth:`expected_lifetime_value`
        — perturbing ``c`` by ``eps`` in the direction that raises the
        average retention rate and measuring the relative change in
        ``E(DRL)``.
        """
        # Baseline lifetime value
        base = self.expected_lifetime_value(
            discount_rate=discount_rate, data=data, max_periods=max_periods
        )

        # Perturb each draw's ``c`` downward by ``eps`` to uniformly raise
        # the retention rate.  (Lower ``c`` → flatter hazard → higher
        # long-run retention.)  The average retention change is
        # approximately the relative change in S(1) at the observed
        # ``T``, which is ``log(1 + d log r)`` to first order in ``eps``.
        from scipy.special import betaln as _betaln_np

        if data is None:
            data = self.data.copy()
        data = data.assign(discount_rate=discount_rate)
        ds = self._extract_predictive_variables(
            data, customer_varnames=["T", "discount_rate", "cohort"]
        )
        alpha, beta, c, T, d = (
            ds["alpha"],
            ds["beta"],
            ds["c"],
            ds["T"],
            ds["discount_rate"],
        )

        ks = xarray.DataArray(np.arange(0, max_periods + 1), dims="k")
        c_perturb = c - eps
        log_cond_surv_pert = _betaln_np(
            alpha, beta + (T + ks) ** c_perturb
        ) - _betaln_np(alpha, beta + T**c_perturb)
        cond_surv_pert = np.exp(log_cond_surv_pert)
        pert = (cond_surv_pert * (1.0 + d) ** (-ks)).sum("k")

        # Base retention rate at tenure T+1
        r_T = np.exp(
            _betaln_np(alpha, beta + (T + 1) ** c) - _betaln_np(alpha, beta + T**c)
        )
        r_T_pert = np.exp(
            _betaln_np(alpha, beta + (T + 1) ** c_perturb)
            - _betaln_np(alpha, beta + T**c_perturb)
        )

        d_edrl = (pert - base) / base
        d_r = (r_T_pert - r_T) / r_T
        elasticity = d_edrl / d_r
        return elasticity.transpose(
            "chain", "draw", "customer_id", "cohort", missing_dims="ignore"
        )

    # ------------------------------------------------------------------
    # Book of Business valuation
    # ------------------------------------------------------------------
    def expected_residual_value(
        self,
        margin: float | np.ndarray | pd.Series = 1.0,
        discount_rate: float | np.ndarray | pd.Series = 0.0,
        data: pd.DataFrame | None = None,
        *,
        max_periods: int = 100,
    ) -> xarray.DataArray:
        r"""Compute expected residual *value* per customer (per-customer RLV).

        This is the discounted expected residual lifetime
        (:meth:`expected_lifetime_value`) multiplied by a per-period
        ``margin`` (contribution, ARPU, or net cash flow per renewal):

        .. math::

            \text{RLV}_i = m_i \cdot \mathbb{E}(DRL_i),

        where :math:`m_i` is the per-period margin for customer ``i``.
        ``margin`` may be a scalar (applied to all customers) or a vector
        aligned with ``data``'s rows for customer-specific economics.

        The result keeps full posterior dimensions
        (``chain``, ``draw``, ``customer_id``, ``cohort``) so downstream
        aggregations propagate parameter uncertainty.
        """
        if data is None:
            data = self.data.copy()

        edrl = self.expected_lifetime_value(
            discount_rate=discount_rate, data=data, max_periods=max_periods
        )

        # Broadcast margin onto the per-customer axis.  After the
        # sBG-mirroring swap_dims in ``_extract_predictive_variables`` the
        # per-customer dimension is labelled ``cohort`` (with ``customer_id``
        # demoted to a coordinate), so detect whichever is present.
        customer_dim = "customer_id" if "customer_id" in edrl.dims else "cohort"
        if np.ndim(margin) == 0:
            margin_da: float | xarray.DataArray = float(margin)
        else:
            margin_values = np.asarray(margin, dtype=float)
            if margin_values.shape[0] != edrl.sizes[customer_dim]:
                raise ValueError(
                    "margin must be a scalar or a vector with one entry per "
                    f"customer (got {margin_values.shape[0]}, expected "
                    f"{edrl.sizes[customer_dim]})."
                )
            margin_da = xarray.DataArray(
                margin_values,
                dims=customer_dim,
                coords={customer_dim: edrl.coords[customer_dim]},
            )

        return edrl * margin_da

    def book_of_business_value(
        self,
        margin: float | np.ndarray | pd.Series = 1.0,
        discount_rate: float | np.ndarray | pd.Series = 0.0,
        data: pd.DataFrame | None = None,
        *,
        max_periods: int = 100,
        by_cohort: bool = False,
    ) -> xarray.DataArray:
        r"""Book of Business: the aggregate residual value of the customer base.

        The Book of Business (BoB) is the sum of expected residual
        lifetime value over every customer who is still active at the end
        of the observation window:

        .. math::

            \text{BoB} = \sum_{i \,:\, \text{alive at } T_i}
                m_i \cdot \mathbb{E}(DRL_i).

        Only customers who are *alive* (``recency == T``) contribute to the
        Book of Business — churned customers have no residual value.  The
        sum is taken for every posterior draw, yielding a full posterior
        distribution over the value of the book (rather than a point
        estimate), which is the whole point of doing this in a Bayesian
        framework: management gets a credible interval on the asset value
        of the customer base.

        Parameters
        ----------
        margin
            Per-period margin per customer.  Scalar or one value per
            *alive* customer.
        discount_rate
            Per-period discount rate applied inside ``E(DRL)``.
        data
            Customer frame.  Defaults to the alive customers in the
            training data.
        max_periods
            Truncation horizon for the residual-lifetime sum.
        by_cohort
            If ``True``, return the Book of Business broken down per
            cohort (keeps the ``cohort`` dim); otherwise return the
            grand total across all alive customers.

        Returns
        -------
        xarray.DataArray
            Posterior of the Book of Business value, with dims
            (``chain``, ``draw``) — or (``chain``, ``draw``, ``cohort``)
            when ``by_cohort=True``.
        """
        if data is None:
            data = self.data.query("recency == T").copy()

        rlv = self.expected_residual_value(
            margin=margin,
            discount_rate=discount_rate,
            data=data,
            max_periods=max_periods,
        )

        if by_cohort:
            if "cohort" in rlv.dims:
                return rlv.sum("customer_id")
            # Cohort dim already collapsed (e.g. single cohort squeezed during
            # prediction); fall through to the grand total, which is the
            # single cohort's value.
        return rlv.sum([d for d in ("customer_id", "cohort") if d in rlv.dims])

    # ------------------------------------------------------------------
    # New-customer population predictions
    # ------------------------------------------------------------------
    def distribution_new_customer_theta(
        self,
        n: int = 1,
        *,
        random_seed: RandomState | None = None,
    ) -> xarray.DataArray:
        """Draw the churn probability ``theta`` for ``n`` new customers."""
        coords = {"new_customer_id": np.arange(n)}
        with pm.Model(coords=coords):
            alpha = pm.Flat("alpha")
            beta = pm.Flat("beta")
            pm.Beta("theta", alpha, beta, dims="new_customer_id")
            return pm.sample_posterior_predictive(
                self.idata,
                var_names=["theta"],
                random_seed=random_seed,
            ).posterior_predictive["theta"]

    def distribution_new_customer_churn_time(
        self,
        n: int = 1,
        *,
        random_seed: RandomState | None = None,
    ) -> xarray.DataArray:
        """Draws of the churn time for ``n`` prospective new customers."""
        coords = {"new_customer_id": np.arange(n)}
        with pm.Model(coords=coords):
            alpha = pm.Flat("alpha")
            beta = pm.Flat("beta")
            c = pm.Flat("c")
            BetaDiscreteWeibull("churn", alpha, beta, c, dims="new_customer_id")
            return pm.sample_posterior_predictive(
                self.idata,
                var_names=["churn"],
                random_seed=random_seed,
            ).posterior_predictive["churn"]


# ---------------------------------------------------------------------------
# Individual-level BdW model (mirrors ShiftedBetaGeoModelIndividual)
# ---------------------------------------------------------------------------


class BetaDiscreteWeibullModelIndividual(CLVModel):
    """Individual-customer-level BdW model with observed churn times.

    Analogous to
    :class:`pymc_marketing.clv.models.ShiftedBetaGeoModelIndividual`, but
    the individual-level lifetime is a :class:`DiscreteWeibull` rather
    than a :class:`pymc.distributions.Geometric`.  Each customer has their
    own ``theta ~ Beta(alpha, beta)``; the shape parameter ``c`` is shared
    across the population by default (matching the paper's
    recommendation, Section 6.2 of [1]_).

    Parameters
    ----------
    data : pandas.DataFrame
        Contains:

        * ``customer_id``: Unique customer identifier.
        * ``t_churn``: Time of cancellation (integer >= 1).  If the customer
          is still active at the end of the observation period, set
          ``t_churn == T`` so the observation is treated as right-censored.
        * ``T``: Maximum observed time period for the customer.

    model_config : dict, optional
        Priors on ``alpha``, ``beta`` and ``c`` — see
        :attr:`default_model_config`.
    sampler_config : dict, optional
        Passed straight through to ``pm.sample``.

    Examples
    --------
    .. code-block:: python

        from pymc_extras.prior import Prior
        from pymc_marketing.clv import BetaDiscreteWeibullModelIndividual

        model = BetaDiscreteWeibullModelIndividual(
            data=df,
            model_config={
                "alpha": Prior("HalfNormal", sigma=10),
                "beta": Prior("HalfStudentT", nu=4, sigma=10),
                "c": Prior("HalfNormal", sigma=1.0),
            },
        )
        model.fit()

    References
    ----------
    .. [1] Fader et al. (2018), §6.2 discusses allowing heterogeneity in
       ``c``; they find negligible empirical support for it over and
       above homogeneous-``c`` BdW.
    """

    _model_type = "Beta-Discrete-Weibull Model (Individual Customers)"

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        *,
        model_config: ModelConfig | None = None,
        sampler_config: dict | None = None,
    ):
        super().__init__(
            data=data, model_config=model_config, sampler_config=sampler_config
        )

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate Beta-Discrete-Weibull Individual-specific data requirements."""
        self._validate_cols(
            data,
            required_cols=["customer_id", "t_churn", "T"],
            must_be_unique=["customer_id"],
        )
        if np.any(
            (data["t_churn"] < 1)
            | (data["t_churn"] > data["T"])
            | np.isnan(data["t_churn"])
        ):
            raise ValueError(
                "t_churn must satisfy 1 <= t_churn <= T.  Customers that "
                "are still alive should have t_churn == T so their record "
                "is treated as right-censored."
            )

    @property
    def default_model_config(self) -> ModelConfig:
        """Default priors for the individual BdW model."""
        return {
            "alpha": Prior("HalfFlat"),
            "beta": Prior("HalfFlat"),
            "c": Prior("HalfNormal", sigma=1.0),
        }

    def build_model(self, data: pd.DataFrame | None = None) -> None:  # type: ignore[override]
        r"""Build the PyMC model with ``theta`` marginalized analytically.

        The hierarchical formulation — one latent
        ``theta_i ~ Beta(alpha, beta)`` per customer feeding a
        ``DiscreteWeibull(theta_i, c)`` lifetime — is mathematically
        equivalent to placing the closed-form Beta mixture
        :class:`BetaDiscreteWeibull` directly on each customer's lifetime:

        .. math::

            \int_0^1 P(T = t \mid \theta, c)\,
                \mathrm{Beta}(\theta \mid \alpha, \beta)\,d\theta
            \;=\; P_{BdW}(T = t \mid \alpha, \beta, c).

        Sampling the marginal form is strongly preferred: the explicit
        formulation puts ~N latent ``theta`` parameters in a funnel with
        ``(alpha, beta)`` and on a curved ridge against ``c``, which
        produces heavy NUTS divergences once ``c`` is free.  The marginal
        model has only the three population-level parameters, samples
        divergence-free, and yields the identical posterior for
        ``(alpha, beta, c)``.

        Per-customer quantities are unaffected: the
        ``distribution_*`` posterior-predictive helpers redraw
        ``theta ~ Beta(alpha, beta)`` from the population posterior, which
        is exactly what they did before this reparameterisation.
        """
        if data is not None:
            self._validate_data(data)
            self.data = data
        elif not hasattr(self, "data") or self.data is None:
            raise ValueError(
                f"{self._model_type}.build_model() requires data parameter. "
                "Either pass data to build_model(data=...) or fit(data=...)"
            )
        else:
            self._validate_data(self.data)

        coords = {"customer_id": self.data["customer_id"]}
        with pm.Model(coords=coords) as self.model:
            alpha = self.model_config["alpha"].create_variable("alpha")
            beta = self.model_config["beta"].create_variable("beta")
            c = self.model_config["c"].create_variable("c")

            # theta is integrated out analytically (see docstring); the
            # per-customer lifetime follows the closed-form Beta mixture.
            churn = BetaDiscreteWeibull.dist(alpha, beta, c)
            pm.Censored(
                "churn_censored",
                churn,
                lower=None,
                upper=self.data["T"],
                observed=self.data["t_churn"],
                dims="customer_id",
            )

    # ----- Predictive / posterior-predictive helpers -----
    def distribution_customer_churn_time(
        self,
        customer_id: np.ndarray | pd.Series,
        *,
        random_seed: RandomState | None = None,
    ) -> xarray.DataArray:
        """Posterior-predictive draws of churn time for existing customers."""
        coords = {"customer_id": np.asarray(customer_id)}
        with pm.Model(coords=coords):
            alpha = pm.Flat("alpha")
            beta = pm.Flat("beta")
            c = pm.Flat("c")
            theta = pm.Beta("theta", alpha, beta, dims="customer_id")
            DiscreteWeibull("churn", theta, c, dims="customer_id")
            return pm.sample_posterior_predictive(
                self.idata,
                var_names=["churn"],
                random_seed=random_seed,
            ).posterior_predictive["churn"]

    def distribution_new_customer_churn_time(
        self,
        n: int = 1,
        *,
        random_seed: RandomState | None = None,
        var_names: Sequence[str] = ("theta", "churn"),
    ) -> xarray.Dataset:
        """Draws of churn time and theta for ``n`` as-yet-unacquired customers."""
        coords = {"new_customer_id": np.arange(n)}
        with pm.Model(coords=coords):
            alpha = pm.Flat("alpha")
            beta = pm.Flat("beta")
            c = pm.Flat("c")
            theta = pm.Beta("theta", alpha, beta, dims="new_customer_id")
            DiscreteWeibull("churn", theta, c, dims="new_customer_id")
            return pm.sample_posterior_predictive(
                self.idata,
                var_names=var_names,
                random_seed=random_seed,
            ).posterior_predictive

    def distribution_new_customer_theta(
        self,
        n: int = 1,
        *,
        random_seed: RandomState | None = None,
    ) -> xarray.DataArray:
        """Draw ``theta`` for ``n`` new customers (population posterior)."""
        return self.distribution_new_customer_churn_time(
            n=n, random_seed=random_seed, var_names=("theta",)
        )["theta"]
