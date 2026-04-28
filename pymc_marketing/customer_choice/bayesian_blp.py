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
"""Bayesian BLP (Berry-Levinsohn-Pakes) model on aggregate market-share panels.

Fits the random-coefficients aggregate-share demand model of Berry, Levinsohn
& Pakes (1995) in a fully Bayesian formulation that follows Jiang, Manchanda
& Rossi (2009, *QME*) and Yang, Chen & Allenby (2003): the BLP contraction
mapping and GMM are dropped in favour of a joint posterior over preference
parameters and the latent product-market shocks ``ξ_jt``. This makes
hierarchical pooling across markets / regions cheap, returns full posterior
elasticities, and stays honest under weak instruments.

Use this when the data is *aggregate market shares* across products and
markets (the common Nielsen / IRI / retailer-scanner data shape) and you
need cross-price substitution patterns that come from a structural
preference model rather than a reduced-form share allocation. For
*individual-level* discrete choice, use
:class:`pymc_marketing.customer_choice.MixedLogit` instead. For
"what happened when X launched" style impact analyses on aggregate shares,
use :class:`pymc_marketing.customer_choice.MVITS`.
"""

import json
import warnings
from typing import Any, Literal

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import xarray as xr
from pymc.util import RandomState
from pymc_extras.prior import Prior

from pymc_marketing.customer_choice._choice_helpers import (
    halton_draws,
    non_centered_normal,
)
from pymc_marketing.model_builder import ModelBuilder, create_sample_kwargs
from pymc_marketing.model_config import parse_model_config
from pymc_marketing.version import __version__

_VALID_LIKELIHOODS = ("normal_logshare",)


class BayesianBLP(ModelBuilder):
    """Bayesian random-coefficients logit on aggregate market-share panels.

    Parameters
    ----------
    market_data : pd.DataFrame
        Long-format panel. Each (region, market, product) cell is one row.
        Every market must contain exactly one row per inside product plus a
        single outside-good row whose ``product_col`` value matches
        ``outside_good``. Outside-good rows should have ``price``,
        characteristics, and instruments all set to ``0``.
    product_col, market_col, region_col, share_col, market_size_col, price_col : str
        Column names. ``region_col=None`` (default) collapses the region
        hierarchy to a single bucket. ``market_col`` must uniquely identify a
        (region, period) cell.
    characteristics : list of str
        Columns holding product characteristics ``x_jt``.
    instruments : list of str, optional
        Columns holding instruments ``z_jt`` for the price-endogeneity
        block. If ``None``, no first-stage price equation is built and the
        price coefficient is *not identified* under endogeneity — a warning
        is raised.
    outside_good : str
        Row label of the outside good in ``product_col``.
    n_mc_draws : int, optional
        Number of Owen-scrambled Halton draws used to integrate the share
        equation over consumer heterogeneity. Defaults to
        ``max(200, 100 * n_random_coefs)`` and warns when the chosen value
        looks too small for the integration dimension.
    random_coef_on : list of str, optional
        Names of dimensions that receive consumer-level random coefficients.
        Use the literal string ``"price"`` for the price coefficient and any
        characteristic name for that characteristic. Defaults to
        ``["price"]``.
    product_fixed_effects : bool
        If ``True`` (default), the structural error decomposes as
        ``ξ_jt = ξ_j + ξ̃_jt`` with a product fixed effect. If ``False``,
        per-product alternative-specific intercepts are used instead and
        ``ξ_jt = ξ̃_jt``. Only one of the two is included, never both.
        ``False`` is not supported in this v1 release; pass ``True``.
    likelihood : {"normal_logshare"}
        Aggregate-share likelihood. Currently only the Berry (1994)
        heteroskedastic Normal-on-log-share-ratio formulation is wired up.
    min_share : float
        Floor applied to observed shares to avoid ``log(0)``. A warning is
        emitted when the floor is hit.
    track_delta : bool
        If ``True``, store the mean-utility tensor ``δ_jt`` as a
        ``pm.Deterministic`` (memory-heavy on large panels). Default
        ``False``.
    model_config, sampler_config : dict, optional
        Standard ``ModelBuilder`` overrides. The default sampler configuration
        targets ``numpyro`` at ``target_accept=0.95`` because the
        ``ξ̃_jt`` block is funnel-prone.

    Notes
    -----
    *Identification.* Endogeneity correction uses the conditional
    decomposition of the joint ``(η_jt, ξ̃_jt)`` Normal: the price equation
    ``p_jt = π_0j + π_z · z_jt + η_jt`` is fit as a marginal likelihood,
    ``η_jt`` is the price residual, and ``ξ̃_jt | η_jt`` follows
    ``N(ρ · (σ_ξ/σ_η) · η_jt, σ_ξ · sqrt(1 − ρ²))``. This is mathematically
    equivalent to a joint MvNormal but admits a cleaner non-centered
    parameterisation and exposes ``ρ`` as a directly-interpretable
    correlation parameter.

    *Sampler geometry.* All large blocks (``ξ̃_jt``, region-level
    hierarchy, the random-coef raw draws) are non-centered. ``target_accept``
    defaults to ``0.95``. Set ``track_delta=True`` only if you actually need
    the per-cell mean utility in the trace — on a typical 100-week × 10-SKU
    panel this is ~7 MB per chain.
    """

    _model_type = "Bayesian BLP"
    version = "0.1.0"

    def __init__(
        self,
        market_data: pd.DataFrame,
        *,
        characteristics: list[str],
        product_col: str = "product",
        market_col: str = "market",
        region_col: str | None = None,
        share_col: str = "share",
        market_size_col: str = "n",
        price_col: str = "price",
        instruments: list[str] | None = None,
        outside_good: str = "outside",
        n_mc_draws: int | None = None,
        random_coef_on: list[str] | None = None,
        product_fixed_effects: bool = True,
        likelihood: str = "normal_logshare",
        min_share: float = 1e-4,
        track_delta: bool = False,
        model_config: dict | None = None,
        sampler_config: dict | None = None,
        random_seed: int | None = None,
    ) -> None:
        if likelihood not in _VALID_LIKELIHOODS:
            raise ValueError(
                f"likelihood must be one of {_VALID_LIKELIHOODS}, got {likelihood!r}"
            )
        if not product_fixed_effects:
            raise NotImplementedError(
                "product_fixed_effects=False is not implemented in v1; "
                "use True (the default)."
            )

        self.market_data = market_data
        self.product_col = product_col
        self.market_col = market_col
        self.region_col = region_col
        self.share_col = share_col
        self.market_size_col = market_size_col
        self.price_col = price_col
        self.characteristics = list(characteristics)
        self.instruments = list(instruments) if instruments else None
        self.outside_good = outside_good
        self.product_fixed_effects = product_fixed_effects
        self.likelihood = likelihood
        self.min_share = float(min_share)
        self.track_delta = bool(track_delta)
        self.random_seed = random_seed

        self._random_coef_on = (
            list(random_coef_on) if random_coef_on is not None else ["price"]
        )
        self._validate_random_coef_on()

        self._preprocess()

        if n_mc_draws is None:
            n_mc_draws = max(200, 100 * max(self._n_random, 1))
        if self._n_random > 4 and n_mc_draws < 500:
            warnings.warn(
                f"n_mc_draws={n_mc_draws} is small for {self._n_random} random "
                "coefficients; consider raising it to >=500.",
                UserWarning,
                stacklevel=2,
            )
        self.n_mc_draws = int(n_mc_draws)

        halton_dim = max(self._n_random, 1)
        self._halton = halton_draws(self.n_mc_draws, halton_dim, seed=random_seed)

        if self.instruments is None:
            warnings.warn(
                "BayesianBLP was constructed with instruments=None. The price "
                "coefficient is not identified under endogeneity; the posterior "
                "on alpha will absorb any price-xi correlation as bias.",
                UserWarning,
                stacklevel=2,
            )

        self._build_coords()

        super().__init__(
            model_config=parse_model_config(model_config or {}),
            sampler_config=sampler_config,
        )

    def _validate_random_coef_on(self) -> None:
        unknown = [
            name
            for name in self._random_coef_on
            if name != "price" and name not in self.characteristics
        ]
        if unknown:
            raise ValueError(
                f"random_coef_on entries {unknown} are not in characteristics "
                f"{self.characteristics} and are not the literal 'price'."
            )
        self._random_on_price = "price" in self._random_coef_on
        self._random_char_names = [c for c in self._random_coef_on if c != "price"]
        self._random_char_idx = [
            self.characteristics.index(c) for c in self._random_char_names
        ]
        self._random_coef_names = (
            ["price"] if self._random_on_price else []
        ) + self._random_char_names
        self._n_random = len(self._random_coef_names)

    def _preprocess(self) -> None:
        df = self.market_data
        required = [self.product_col, self.market_col, self.share_col, self.price_col]
        if self.market_size_col is not None:
            required.append(self.market_size_col)
        if self.region_col is not None:
            required.append(self.region_col)
        for col in [*required, *self.characteristics, *(self.instruments or [])]:
            if col not in df.columns:
                raise ValueError(f"Column {col!r} not found in market_data.")

        if self.outside_good not in df[self.product_col].unique():
            raise ValueError(
                f"outside_good={self.outside_good!r} not present in "
                f"{self.product_col!r} column."
            )

        inside_products = [
            p for p in df[self.product_col].unique() if p != self.outside_good
        ]
        inside_products = sorted(inside_products)
        self._inside_products: list[str] = inside_products
        J = len(inside_products)

        markets = df[self.market_col].unique().tolist()
        M = len(markets)
        market_to_idx = {m: i for i, m in enumerate(markets)}
        self._markets: list[Any] = markets

        product_counts = df.groupby(self.market_col)[self.product_col].nunique()
        if not (product_counts == J + 1).all():
            offenders = product_counts[product_counts != J + 1].index.tolist()
            raise ValueError(
                f"Every market must contain {J + 1} product rows "
                f"({J} inside + 1 outside). Offending markets: {offenders[:5]}"
            )

        if self.region_col is None:
            self._regions: list[str] = ["all"]
            region_idx = np.zeros(M, dtype=int)
        else:
            region_per_market = (
                df.drop_duplicates(self.market_col)
                .set_index(self.market_col)[self.region_col]
                .reindex(markets)
            )
            regions = sorted(region_per_market.unique().tolist())
            self._regions = regions
            region_to_idx = {r: i for i, r in enumerate(regions)}
            region_idx = np.array(
                [region_to_idx[r] for r in region_per_market.values], dtype=int
            )

        self._region_idx = region_idx

        K = len(self.characteristics)
        L = len(self.instruments) if self.instruments else 0

        inside_share = np.zeros((M, J))
        outside_share = np.zeros(M)
        n = np.zeros(M)
        price = np.zeros((M, J))
        x = np.zeros((M, J, K))
        z = np.zeros((M, J, L)) if L > 0 else None

        product_to_idx = {p: i for i, p in enumerate(inside_products)}

        for _, row in df.iterrows():
            m_idx = market_to_idx[row[self.market_col]]
            n[m_idx] = row[self.market_size_col]
            if row[self.product_col] == self.outside_good:
                outside_share[m_idx] = row[self.share_col]
            else:
                j_idx = product_to_idx[row[self.product_col]]
                inside_share[m_idx, j_idx] = row[self.share_col]
                price[m_idx, j_idx] = row[self.price_col]
                for k, col in enumerate(self.characteristics):
                    x[m_idx, j_idx, k] = row[col]
                if z is not None:
                    for ell, col in enumerate(self.instruments or []):
                        z[m_idx, j_idx, ell] = row[col]

        if self.min_share > 0:
            floored_inside = np.maximum(inside_share, self.min_share)
            floored_outside = np.maximum(outside_share, self.min_share)
            n_floored = int((floored_inside != inside_share).sum()) + int(
                (floored_outside != outside_share).sum()
            )
            if n_floored:
                warnings.warn(
                    f"Floored {n_floored} share value(s) below min_share="
                    f"{self.min_share}; observed shares were near zero. "
                    "Consider raising min_share or excluding sparse cells.",
                    UserWarning,
                    stacklevel=3,
                )
            inside_share = floored_inside
            outside_share = floored_outside

        self._inside_share = inside_share
        self._outside_share = outside_share
        self._n = n
        self._price = price
        self._x = x
        self._z = z
        self._M = M
        self._J = J
        self._K = K
        self._L = L

    def _build_coords(self) -> None:
        coords: dict[str, list] = {
            "market": list(self._markets),
            "region": list(self._regions),
            "inside_product": list(self._inside_products),
            "characteristic": list(self.characteristics),
            "mc_draw": list(range(self.n_mc_draws)),
        }
        if self.instruments:
            coords["instrument"] = list(self.instruments)
        if self._n_random > 0:
            coords["random_coef"] = list(self._random_coef_names)
        self.coords = coords

    @property
    def default_model_config(self) -> dict:
        """Default priors for every univariate / vector parameter in the model.

        Joint blocks (the price-xi correlation, non-centered raw normals)
        are constructed inline in ``build_model`` because they don't slot
        into the ``Prior`` abstraction cleanly. Hierarchical entries
        (``tau_alpha``, ``tau_beta``) are only consumed when ``region_col``
        is set; ``sigma_random`` only when at least one random coefficient
        is requested; first-stage entries (``pi_0``, ``pi_z``,
        ``sigma_eta``) only when instruments are supplied.
        """
        config = {
            "alpha": Prior("Normal", mu=0.0, sigma=2.0),
            "beta": Prior("Normal", mu=0.0, sigma=1.0, dims="characteristic"),
            "tau_alpha": Prior("HalfNormal", sigma=1.0),
            "tau_beta": Prior("HalfNormal", sigma=1.0, dims="characteristic"),
            "sigma_xi": Prior("HalfNormal", sigma=1.0),
            "sigma_xi_j": Prior("HalfNormal", sigma=0.5),
        }
        if self._n_random > 0:
            config["sigma_random"] = Prior("HalfNormal", sigma=1.0, dims="random_coef")
        if self.instruments:
            config["pi_0"] = Prior("Normal", mu=2.0, sigma=1.0, dims="inside_product")
            config["pi_z"] = Prior("Normal", mu=0.0, sigma=1.0, dims="instrument")
            config["sigma_eta"] = Prior("HalfNormal", sigma=1.0)
        return config

    @property
    def default_sampler_config(self) -> dict:
        """Default sampler kwargs: ``numpyro`` NUTS at ``target_accept=0.95``.

        The high target-accept reflects the funnel-prone hierarchical and
        ``ξ̃_jt`` blocks. Override at construction or pass kwargs to
        ``fit`` for one-off changes.
        """
        return {
            "nuts_sampler": "numpyro",
            "target_accept": 0.95,
            "draws": 2000,
            "tune": 2000,
            "chains": 4,
            "idata_kwargs": {"log_likelihood": True},
        }

    @property
    def output_var(self) -> str:
        """Name of the observed variable (the log-share-ratio likelihood)."""
        return "log_share_ratio"

    @property
    def _serializable_model_config(self) -> dict[str, Any]:
        return {k: v.to_dict() for k, v in self.model_config.items()}

    def _make_region_coefs(self) -> tuple[pt.TensorVariable, pt.TensorVariable]:
        if self.region_col is None:
            alpha_scalar = self.model_config["alpha"].create_variable("alpha")
            beta_scalar = self.model_config["beta"].create_variable("beta")
            alpha_r = pm.Deterministic(
                "alpha_r", pt.atleast_1d(alpha_scalar), dims="region"
            )
            beta_r = pm.Deterministic(
                "beta_r", beta_scalar[None, :], dims=("region", "characteristic")
            )
            return alpha_r, beta_r

        alpha_pop = self.model_config["alpha"].create_variable("alpha_pop")
        beta_pop = self.model_config["beta"].create_variable("beta_pop")
        tau_alpha = self.model_config["tau_alpha"].create_variable("tau_alpha")
        tau_beta = self.model_config["tau_beta"].create_variable("tau_beta")

        alpha_r = non_centered_normal("alpha_r", alpha_pop, tau_alpha, dims="region")
        beta_r_raw = pm.Normal(
            "beta_r_raw", 0.0, 1.0, dims=("region", "characteristic")
        )
        beta_r = pm.Deterministic(
            "beta_r",
            beta_pop[None, :] + tau_beta[None, :] * beta_r_raw,
            dims=("region", "characteristic"),
        )
        return alpha_r, beta_r

    def _make_xi_block(
        self, price_data: pt.TensorVariable, z_data: pt.TensorVariable | None
    ) -> tuple[pt.TensorVariable, pt.TensorVariable | None]:
        sigma_xi = self.model_config["sigma_xi"].create_variable("sigma_xi")
        sigma_xi_j = self.model_config["sigma_xi_j"].create_variable("sigma_xi_j")

        xi_j = non_centered_normal("xi_j", 0.0, sigma_xi_j, dims="inside_product")

        if z_data is None:
            xi_tilde = non_centered_normal(
                "xi_tilde", 0.0, sigma_xi, dims=("market", "inside_product")
            )
            eta = None
        else:
            sigma_eta = self.model_config["sigma_eta"].create_variable("sigma_eta")
            pi_0 = self.model_config["pi_0"].create_variable("pi_0")
            pi_z = self.model_config["pi_z"].create_variable("pi_z")
            mu_p = pi_0[None, :] + pt.sum(z_data * pi_z[None, None, :], axis=2)

            pm.Normal(
                "price_obs",
                mu=mu_p,
                sigma=sigma_eta,
                observed=price_data,
                dims=("market", "inside_product"),
            )

            eta = pm.Deterministic(
                "eta", price_data - mu_p, dims=("market", "inside_product")
            )

            rho = pm.Uniform("rho_price_xi", lower=-0.99, upper=0.99)
            sigma_xi_cond = sigma_xi * pt.sqrt(1.0 - rho * rho)
            mu_xi_cond = rho * (sigma_xi / sigma_eta) * eta

            xi_tilde_raw = pm.Normal(
                "xi_tilde_raw", 0.0, 1.0, dims=("market", "inside_product")
            )
            xi_tilde = pm.Deterministic(
                "xi_tilde",
                mu_xi_cond + sigma_xi_cond * xi_tilde_raw,
                dims=("market", "inside_product"),
            )

        xi = pm.Deterministic(
            "xi", xi_j[None, :] + xi_tilde, dims=("market", "inside_product")
        )
        return xi, eta

    def _make_mu_dev(
        self,
        price_data: pt.TensorVariable,
        x_data: pt.TensorVariable,
        halton_data: pt.TensorVariable,
    ) -> pt.TensorVariable | None:
        if self._n_random == 0:
            return None
        sigma_random = self.model_config["sigma_random"].create_variable("sigma_random")
        contributions = []
        for d, name in enumerate(self._random_coef_names):
            covariate_jt = (
                price_data
                if name == "price"
                else x_data[..., self.characteristics.index(name)]
            )
            term = (
                sigma_random[d]
                * halton_data[None, None, :, d]
                * covariate_jt[..., None]
            )
            contributions.append(term)
        return sum(contributions[1:], contributions[0])

    def _make_predicted_shares(
        self, delta: pt.TensorVariable, mu_dev: pt.TensorVariable | None
    ) -> tuple[pt.TensorVariable, pt.TensorVariable]:
        if mu_dev is None:
            U = delta[..., None]
        else:
            U = delta[..., None] + mu_dev
        U_max = pt.maximum(U.max(axis=1, keepdims=True), 0.0)
        eU = pt.exp(U - U_max)
        e0 = pt.exp(-U_max[:, 0, :])
        denom = e0 + eU.sum(axis=1)
        s_inside_per_draw = eU / denom[:, None, :]
        s_outside_per_draw = e0 / denom
        s_inside = pm.Deterministic(
            "s_inside",
            s_inside_per_draw.mean(axis=-1),
            dims=("market", "inside_product"),
        )
        s_outside = pm.Deterministic(
            "s_outside", s_outside_per_draw.mean(axis=-1), dims="market"
        )
        return s_inside, s_outside

    def build_model(self, **kwargs) -> None:
        """Construct the PyMC model and attach it to ``self.model``."""
        with pm.Model(coords=self.coords) as model:
            price_data = pm.Data(
                "price", self._price, dims=("market", "inside_product")
            )
            x_data = pm.Data(
                "x", self._x, dims=("market", "inside_product", "characteristic")
            )
            n_data = pm.Data("n", self._n, dims="market")
            inside_share_data = pm.Data(
                "inside_share",
                self._inside_share,
                dims=("market", "inside_product"),
            )
            outside_share_data = pm.Data(
                "outside_share", self._outside_share, dims="market"
            )
            z_data = (
                pm.Data(
                    "z",
                    self._z,
                    dims=("market", "inside_product", "instrument"),
                )
                if self._z is not None
                else None
            )
            halton_data = pm.Data(
                "halton",
                self._halton,
                dims=(
                    ("mc_draw", "random_coef")
                    if self._n_random > 0
                    else ("mc_draw", "halton_dim")
                ),
            )

            alpha_r, beta_r = self._make_region_coefs()
            xi, _eta = self._make_xi_block(price_data, z_data)

            region_idx = pt.constant(self._region_idx, dtype="int64")
            alpha_at_market = alpha_r[region_idx]
            beta_at_market = beta_r[region_idx]

            delta_expr = (
                alpha_at_market[:, None] * price_data
                + pt.sum(x_data * beta_at_market[:, None, :], axis=2)
                + xi
            )
            if self.track_delta:
                delta = pm.Deterministic(
                    "delta", delta_expr, dims=("market", "inside_product")
                )
            else:
                delta = delta_expr

            mu_dev = self._make_mu_dev(price_data, x_data, halton_data)
            s_inside, s_outside = self._make_predicted_shares(delta, mu_dev)

            log_share_ratio_obs = (
                pt.log(inside_share_data) - pt.log(outside_share_data)[:, None]
            )
            log_share_ratio_pred = pt.log(s_inside) - pt.log(s_outside)[:, None]
            sigma2 = (1.0 - inside_share_data) / (n_data[:, None] * inside_share_data)
            sigma_obs = pt.sqrt(sigma2)

            pm.Normal(
                "log_share_ratio",
                mu=log_share_ratio_pred,
                sigma=sigma_obs,
                observed=log_share_ratio_obs,
                dims=("market", "inside_product"),
            )

        self.model = model

    def sample_prior_predictive(
        self,
        samples: int = 500,
        extend_idata: bool = True,
        **kwargs,
    ) -> az.InferenceData:
        """Draw from the prior predictive distribution."""
        if not hasattr(self, "model"):
            self.build_model()
        with self.model:
            prior_pred = pm.sample_prior_predictive(samples, **kwargs)
            prior_pred["prior"].attrs["pymc_marketing_version"] = __version__
            prior_pred["prior_predictive"].attrs["pymc_marketing_version"] = __version__
            self.set_idata_attrs(prior_pred)
        if extend_idata:
            if self.idata is not None:
                self.idata.extend(prior_pred, join="right")
            else:
                self.idata = prior_pred
        return prior_pred

    def fit(
        self,
        progressbar: bool | None = None,
        random_seed: RandomState | None = None,
        **kwargs,
    ) -> az.InferenceData:
        """Fit by sampling the joint posterior with NUTS."""
        if not hasattr(self, "model"):
            self.build_model()
        sampler_kwargs = create_sample_kwargs(
            self.sampler_config, progressbar, random_seed, **kwargs
        )
        with self.model:
            idata = pm.sample(**sampler_kwargs)
            idata.attrs["pymc_marketing_version"] = __version__
            self.set_idata_attrs(idata)
        if self.idata is None:
            self.idata = idata
        else:
            self.idata.extend(idata, join="right")
        self.is_fitted_ = True
        return self.idata

    def build_from_idata(self, idata: az.InferenceData) -> None:
        """Reconstruct the model graph from a saved InferenceData.

        Save / load is wired to the standard ``ModelBuilder`` round-trip in a
        follow-up step; this stub simply rebuilds the model from current
        in-memory data so that posterior predictive sampling on a
        re-loaded class instance works.
        """
        self.build_model()
        self.idata = idata

    def _resolve_price_change(
        self, price_change: dict[str, float] | np.ndarray | None
    ) -> np.ndarray:
        """Translate a user-facing ``price_change`` spec into a (M, J) array.

        ``None`` or empty dict → baseline price (no-op canary).
        ``dict {product: relative_change}`` → multiply baseline price for each
        named product by ``(1 + relative_change)`` across all markets.
        ``np.ndarray`` of shape ``(M, J)`` → use as the new price matrix
        verbatim.
        """
        if price_change is None or (
            isinstance(price_change, dict) and len(price_change) == 0
        ):
            return self._price.copy()
        if isinstance(price_change, np.ndarray):
            if price_change.shape != self._price.shape:
                raise ValueError(
                    f"price_change array shape {price_change.shape} "
                    f"does not match (M, J) = {self._price.shape}"
                )
            return price_change.astype(float, copy=True)
        if isinstance(price_change, dict):
            new_price = self._price.copy()
            for prod_name, rel_change in price_change.items():
                if prod_name not in self._inside_products:
                    raise ValueError(
                        f"Unknown product {prod_name!r}. Known inside "
                        f"products: {self._inside_products}"
                    )
                j = self._inside_products.index(prod_name)
                new_price[:, j] = new_price[:, j] * (1.0 + float(rel_change))
            return new_price
        raise TypeError(
            f"price_change must be dict, ndarray, or None; got {type(price_change)!r}"
        )

    def _iterate_posterior_samples(
        self, n_samples: int | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Stack chain × draw and (optionally) subsample posterior arrays.

        Returns
        -------
        alpha_at_market : (S, M)
        beta_at_market : (S, M, K)
        xi : (S, M, J)
        sigma_random : (S, n_random) or None
        """
        if self.idata is None or "posterior" not in self.idata:
            raise RuntimeError(
                "Model has no posterior; call .fit(...) before "
                "elasticities() or counterfactual_shares()."
            )
        post = self.idata.posterior.stack(sample=("chain", "draw"))
        S_total = post.sizes["sample"]
        if n_samples is not None and n_samples < S_total:
            rng = np.random.default_rng(self.random_seed)
            idx = rng.choice(S_total, size=n_samples, replace=False)
            post = post.isel(sample=idx)

        alpha_r = post["alpha_r"].transpose("sample", "region").values
        beta_r = post["beta_r"].transpose("sample", "region", "characteristic").values
        xi = post["xi"].transpose("sample", "market", "inside_product").values

        alpha_at_market = alpha_r[:, self._region_idx]
        beta_at_market = beta_r[:, self._region_idx, :]

        if self._n_random > 0:
            sigma_random = (
                post["sigma_random"].transpose("sample", "random_coef").values
            )
        else:
            sigma_random = None

        return alpha_at_market, beta_at_market, xi, sigma_random

    def _per_sample_shares(
        self,
        alpha_at_market_s: np.ndarray,
        beta_at_market_s: np.ndarray,
        xi_s: np.ndarray,
        sigma_random_s: np.ndarray | None,
        price: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Numpy-evaluate the share equation for one posterior sample.

        Parameters
        ----------
        alpha_at_market_s : (M,)
        beta_at_market_s : (M, K)
        xi_s : (M, J)
        sigma_random_s : (n_random,) or None
        price : (M, J)

        Returns
        -------
        s_inside_per_draw : (M, J, R)
        s_inside_agg : (M, J)
        s_outside_per_draw : (M, R)
        s_outside_agg : (M,)
        alpha_per_draw : (M, R)  -- per-consumer-draw price coefficient
        """
        M, J = self._M, self._J
        R = self.n_mc_draws

        delta = (
            alpha_at_market_s[:, None] * price
            + np.einsum("mjk,mk->mj", self._x, beta_at_market_s)
            + xi_s
        )

        if sigma_random_s is not None and self._n_random > 0:
            mu_dev = np.zeros((M, J, R))
            for d, name in enumerate(self._random_coef_names):
                if name == "price":
                    cov_jt = price
                else:
                    cov_jt = self._x[..., self.characteristics.index(name)]
                mu_dev = mu_dev + (
                    sigma_random_s[d]
                    * self._halton[None, None, :, d]
                    * cov_jt[..., None]
                )
        else:
            mu_dev = np.zeros((M, J, 1))

        U = delta[..., None] + mu_dev
        U_max = np.maximum(U.max(axis=1, keepdims=True), 0.0)
        eU = np.exp(U - U_max)
        e0 = np.exp(-U_max[:, 0, :])
        denom = e0 + eU.sum(axis=1)
        s_inside_per_draw = eU / denom[:, None, :]
        s_outside_per_draw = e0 / denom

        if self._random_on_price and sigma_random_s is not None:
            d_price = self._random_coef_names.index("price")
            alpha_per_draw = (
                alpha_at_market_s[:, None]
                + sigma_random_s[d_price] * self._halton[None, :, d_price]
            )
        else:
            r_eff = mu_dev.shape[-1]
            alpha_per_draw = np.broadcast_to(
                alpha_at_market_s[:, None], (M, r_eff)
            ).copy()

        return (
            s_inside_per_draw,
            s_inside_per_draw.mean(axis=-1),
            s_outside_per_draw,
            s_outside_per_draw.mean(axis=-1),
            alpha_per_draw,
        )

    def elasticities(
        self,
        *,
        at: Literal["mean", "samples"] = "mean",
        n_samples: int = 200,
    ) -> xr.DataArray:
        """Posterior price elasticities ``ε[market, share, price]``.

        Computes the closed-form mixed-logit elasticity from posterior draws::

            ε_jk(m) = (p_km / s_jm) · (1/R) Σ_r α_ir(m)
                       · s_jmr · (δ_jk − s_kmr)

        which is negative on the diagonal (own-price) and positive
        off-diagonal (cross-price substitutes). The integral over consumer
        types is approximated with the same Halton draws used for the
        likelihood, so it is essentially free.

        Parameters
        ----------
        at : {"mean", "samples"}
            ``"mean"`` (default) returns the posterior mean elasticity per
            cell, dims ``(market, share, price)``. ``"samples"`` returns the
            full per-sample array, dims ``(sample, market, share, price)``.
        n_samples : int
            Number of posterior samples to use. With ``at="mean"`` the
            samples are averaged after computation (correct for Jensen's
            inequality, unlike plugging in posterior-mean parameters);
            with ``at="samples"`` they are returned as-is.

        Returns
        -------
        xr.DataArray
        """
        alpha_M, beta_M, xi_M, sigma_M = self._iterate_posterior_samples(n_samples)
        S = alpha_M.shape[0]
        M, J = self._M, self._J
        elast = np.empty((S, M, J, J))
        for i in range(S):
            sigma_i = sigma_M[i] if sigma_M is not None else None
            s_per_draw, s_agg, _, _, alpha_per_draw = self._per_sample_shares(
                alpha_M[i], beta_M[i], xi_M[i], sigma_i, self._price
            )
            R_eff = s_per_draw.shape[-1]
            weighted = alpha_per_draw[:, None, :] * s_per_draw
            ds_dp = -np.einsum("mjr,mkr->mjk", weighted, s_per_draw) / R_eff
            diag_add = weighted.mean(axis=-1)
            for j in range(J):
                ds_dp[:, j, j] += diag_add[:, j]
            elast[i] = ds_dp * self._price[:, None, :] / s_agg[:, :, None]

        da = xr.DataArray(
            elast,
            dims=("sample", "market", "share", "price"),
            coords={
                "market": list(self._markets),
                "share": list(self._inside_products),
                "price": list(self._inside_products),
            },
            name="elasticity",
        )
        if at == "mean":
            return da.mean(dim="sample")
        if at == "samples":
            return da
        raise ValueError(f"at must be 'mean' or 'samples', got {at!r}")

    def counterfactual_shares(
        self,
        price_change: dict[str, float] | np.ndarray | None = None,
        *,
        n_samples: int | None = 200,
    ) -> xr.Dataset:
        """Posterior shares under a counterfactual price intervention.

        Holds the posterior ``ξ_jt`` (the structural shock) fixed and
        recomputes the share equation at the new price. This is the
        BLP-correct *structural* counterfactual: "what would shares have
        been if price for product X had been Y, given the realised demand
        shocks?".

        Notes
        -----
        We do **not** route this through ``pm.do`` + ``sample_posterior_predictive``.
        Because the model uses the conditional decomposition
        ``ξ̃ = ρ·(σ_ξ/σ_η)·η + σ_ξ·sqrt(1-ρ²)·raw`` for sampler geometry,
        ``ξ`` would silently shift when price changes (since
        ``η = price - μ_p`` and the conditional mean of ``ξ̃`` depends on
        ``η``). The numpy path used here pulls the assembled ``xi``
        Deterministic from the posterior and treats it as fixed, which
        gives the structural elasticity semantics.

        Parameters
        ----------
        price_change : dict, ndarray, or None
            ``dict {product_name: relative_change}`` — multiplicative shift
            applied across all markets (e.g. ``{"sku_a": 0.10}`` raises
            ``sku_a`` price by 10%). ``ndarray`` of shape ``(M, J)`` — full
            replacement price matrix. ``None`` — no change (canary; should
            reproduce fitted shares).
        n_samples : int, optional
            Number of posterior samples to draw the counterfactual at.
            ``None`` uses every chain × draw.

        Returns
        -------
        xr.Dataset with ``s_inside (sample, market, inside_product)`` and
        ``s_outside (sample, market)``.
        """
        new_price = self._resolve_price_change(price_change)
        alpha_M, beta_M, xi_M, sigma_M = self._iterate_posterior_samples(n_samples)
        S = alpha_M.shape[0]
        M, J = self._M, self._J
        s_inside_arr = np.empty((S, M, J))
        s_outside_arr = np.empty((S, M))
        for i in range(S):
            sigma_i = sigma_M[i] if sigma_M is not None else None
            _, s_inside_arr[i], _, s_outside_arr[i], _ = self._per_sample_shares(
                alpha_M[i], beta_M[i], xi_M[i], sigma_i, new_price
            )

        return xr.Dataset(
            {
                "s_inside": (("sample", "market", "inside_product"), s_inside_arr),
                "s_outside": (("sample", "market"), s_outside_arr),
            },
            coords={
                "market": list(self._markets),
                "inside_product": list(self._inside_products),
                "sample": np.arange(S),
            },
        )

    def create_idata_attrs(self) -> dict[str, str]:
        """Serialise constructor arguments onto ``InferenceData.attrs``.

        Every ``__init__`` parameter is registered so the base
        ``ModelBuilder.set_idata_attrs`` invariant holds. Heavy or
        non-JSON-serialisable arguments (``market_data``) are stored as
        placeholder strings; full save/load round-tripping is wired up in a
        follow-up step.
        """
        attrs = super().create_idata_attrs()
        attrs["market_data"] = json.dumps("placeholder for market_data DataFrame")
        attrs["product_col"] = json.dumps(self.product_col)
        attrs["market_col"] = json.dumps(self.market_col)
        attrs["region_col"] = json.dumps(self.region_col)
        attrs["share_col"] = json.dumps(self.share_col)
        attrs["market_size_col"] = json.dumps(self.market_size_col)
        attrs["price_col"] = json.dumps(self.price_col)
        attrs["characteristics"] = json.dumps(self.characteristics)
        attrs["instruments"] = json.dumps(self.instruments)
        attrs["outside_good"] = json.dumps(self.outside_good)
        attrs["n_mc_draws"] = json.dumps(self.n_mc_draws)
        attrs["random_coef_on"] = json.dumps(self._random_coef_on)
        attrs["product_fixed_effects"] = json.dumps(self.product_fixed_effects)
        attrs["likelihood"] = json.dumps(self.likelihood)
        attrs["min_share"] = json.dumps(self.min_share)
        attrs["track_delta"] = json.dumps(self.track_delta)
        attrs["random_seed"] = json.dumps(self.random_seed)
        return attrs
