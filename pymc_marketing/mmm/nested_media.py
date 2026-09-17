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
"""Media effect at a finer granularity, hierarchically pooled under a parent.

``NestedMediaEffect`` nests one media granularity inside another: child
units (campaigns, ad sets, creatives, publishers, regions) each get a curve
tied to their parent's curve by size. The docs below use campaigns nested in
channels, the first use case; the structure is the same for any pair of
levels. It gives an MMM a short-term directional signal at
campaign level without destabilising channel-level ROI.  Campaign-level
identification comes from per-campaign saturation (the mix matters because
``sum(f(x_c)) != f(sum(x_c))``), extra model dims, and lift tests — combined
with hierarchical pooling of campaign parameters under channel hyperpriors.

Design notes
------------
- The effect owns its data variable (``campaign_data`` with dims
  ``("date", *mmm.dims, child_dim)``) and is meant to *replace* the
  built-in channel media term for the channels it covers.  Channel-level
  contributions are recovered as ``f"{prefix}_{parent_dim}_contribution"``.
- Campaign spend is scaled by the *channel* total (max over dates), not per
  campaign, so priors mean the same thing for every campaign in a channel.
- Any :class:`~pymc_marketing.mmm.components.saturation.SaturationTransformation`
  can be used (Michaelis-Menten by default). Its priors live at *channel*
  level and are gathered to campaigns through the parent index; its own
  amplitude parameter is the channel capacity.
- The campaign curve is the channel curve scaled by campaign size on both
  axes::

      contribution_c(x) = cap_c**rho * mult_c * S(x / (cap_c**rho * scale_mult_c))

  where ``cap_c`` is the campaign's max channel-scaled spend and ``S`` the
  shared channel curve.  This degree-1 homogeneity makes the parameterisation
  *split-invariant* (splitting a campaign into parts with the same total
  spend leaves the channel contribution unchanged) and makes marginal
  returns equal across campaigns at proportional spend — for any saturation
  shape.  Deviations from that neutral point must be earned from data
  (flighting contrasts, covariates, lift tests), not from the
  parameterisation.
- Campaign multipliers are non-centred under channel pooling scales and, by
  default, constrained to a spend-share-weighted zero sum within each
  channel.
- ``incrementality_spec`` stays ``None``: ``channel_data`` is never an
  ancestor of this effect's contribution.  Point the budget optimizer at the
  effect via ``BudgetOptimizer(spend_vars=["campaign_data"],
  response_variable="total_response_original_scale")``; the default response
  variable does not depend on ``campaign_data`` and raises at construction.
- Adstock is not yet applied (identification of the campaign split rests on
  saturation, not carryover); pre-transform the data if carryover is needed.
"""

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pymc as pm
import pymc.dims as pmd
import pytensor.xtensor as ptx
import xarray as xr
from pydantic import Field, InstanceOf
from pytensor.xtensor.type import XTensorVariable

from pymc_marketing.mmm.additive_effect import DataVarMuEffect, Model
from pymc_marketing.mmm.components.saturation import (
    MichaelisMentenSaturation,
    SaturationTransformation,
)
from pymc_marketing.mmm.distributions import DimWeightedZeroSumNormal
from pymc_marketing.mmm.lift_test import add_saturation_observations
from pymc_marketing.serialization import serialization

# Which function parameter plays the amplitude role, per saturation class.
# Used only to expose the f"{prefix}_beta_{child_dim}" deterministic; shapes
# not listed here still work, they just skip that convenience deterministic.
_AMPLITUDE_PARAM: dict[str, str] = {
    "MichaelisMentenSaturation": "alpha",
    "LogisticSaturation": "beta",
    "InverseScaledLogisticSaturation": "beta",
    "TanhSaturationBaselined": "beta",
    "HillSaturation": "beta",
    "HillSaturationSigmoid": "beta",
    "RootSaturation": "beta",
    "LogSaturation": "beta",
    "NoSaturation": "beta",
}


def lognormal_relative_lift(
    name: str,
    mu: XTensorVariable,
    sigma: XTensorVariable,
    observed: XTensorVariable,
) -> XTensorVariable:
    """LogNormal lift likelihood with the noise as relative error on the measurement.

    The location is the log of the model's estimated lift ``mu``. The
    log-space scale is derived from the measurement's coefficient of
    variation ``sigma / observed``, a fixed number, so an estimated lift
    that collapses towards zero is penalised quadratically in log space. A
    LogNormal or Gamma moment-matched to ``mu`` and ``sigma`` instead lets
    the spread grow as ``mu`` shrinks and barely penalises that collapse,
    which gives the posterior a degenerate mode.

    The likelihood is median-matched: its median is ``mu`` and its mean is
    ``mu * exp(sigma_log**2 / 2)``, a ``+0.5%`` bias at a 10% coefficient of
    variation. Rows must satisfy the lift-table contract checked by the hook.
    """
    sigma_log = pmd.math.sqrt(pmd.math.log1p((sigma / observed) ** 2))
    return pmd.LogNormal(name, mu=pmd.math.log(mu), sigma=sigma_log, observed=observed)


def _check_lift_rows(df_lift_test: pd.DataFrame) -> None:
    """Enforce the lift-table contract: finite values, ``sigma > 0``, nonzero deltas.

    Rows outside this contract cannot be scored by a lift likelihood and
    would only surface as a ``-inf`` model logp at ``sample()`` time with
    nothing pointing at the offending row.
    """
    cols = df_lift_test[["x", "delta_x", "delta_y", "sigma"]]
    bad = (
        ~np.isfinite(cols).all(axis=1)
        | (cols["sigma"] <= 0)
        | (cols["delta_x"] == 0)
        | (cols["delta_y"] == 0)
    )
    if bad.any():
        raise ValueError(
            "df_lift_test rows must have finite x, delta_x, delta_y and sigma, "
            "with sigma > 0 and nonzero delta_x and delta_y; offending rows: "
            f"{df_lift_test.index[bad].tolist()}"
        )


def _channel_scales(
    spend: xr.DataArray, channel_of: xr.DataArray, child_dim: str
) -> tuple[xr.DataArray, xr.DataArray]:
    """Channel scale and campaign cap from the spend array.

    ``channel_of`` labels each campaign with its channel and is named after
    the channel coordinate. The scale of a channel is the max over dates (and
    any extra model dims) of its summed campaign spend; the cap of a campaign
    is the max of its spend on that scaled axis. Channels with no spend get
    scale 1 so nothing divides by zero.
    """
    reduce_dims = [d for d in spend.dims if d != child_dim]
    parent_dim = str(channel_of.name)
    channel_total = spend.groupby(channel_of).sum(child_dim)
    scale = channel_total.max(reduce_dims)
    scale = scale.where(scale > 0, 1.0)
    scale_of_campaign = scale.sel({parent_dim: channel_of}).drop_vars(parent_dim)
    cap = (spend / scale_of_campaign).max(reduce_dims)
    return scale, cap


def _spend_shares(
    total_spend: xr.DataArray, channel_of: xr.DataArray, child_dim: str
) -> xr.DataArray:
    """Each campaign's share of its channel's total spend; zero when the channel has none."""
    parent_dim = str(channel_of.name)
    channel_total = total_spend.groupby(channel_of).sum(child_dim)
    channel_total = channel_total.where(channel_total > 0, 1.0)
    return total_spend / channel_total.sel({parent_dim: channel_of}).drop_vars(
        parent_dim
    )


def _centred_covariates(
    cov: xr.DataArray, share: xr.DataArray, channel_of: xr.DataArray, child_dim: str
) -> xr.DataArray:
    """Centre campaign covariates on their channel's spend-share-weighted mean.

    The covariate term then reallocates efficiency between a channel's
    campaigns but cannot move the channel total. Campaigns of a channel with
    no spend have share zero and keep the raw covariate.
    """
    parent_dim = str(channel_of.name)
    channel_mean = (share * cov).groupby(channel_of).sum(child_dim)
    return cov - channel_mean.sel({parent_dim: channel_of}).drop_vars(parent_dim)


class NestedMediaEffect(DataVarMuEffect):
    """Media effect at a child granularity, hierarchically pooled by parent.

    Campaigns nested in channels are the running example; any child level
    nested in any parent level works the same way, with the child dimension
    named by ``child_dim`` and the parent by ``parent_dim``. Registered
    variable names are built from those dimension names, so the defaults
    give ``nested_media_campaign_contribution`` and
    ``nested_media_channel_contribution``.

    Parameters
    ----------
    child_to_parent : dict[str, str]
        Maps each child (campaign) name to its parent (channel) name.  Must
        cover exactly the children present in the data variable's child
        coordinate.  Ragged parents (different child counts) are fine.
    saturation : SaturationTransformation, optional
        Any saturation from :mod:`pymc_marketing.mmm.components.saturation`;
        Michaelis-Menten by default.  Its priors are created at *channel*
        level (dims default to the effect's channel coordinate) and gathered
        to campaigns; its amplitude parameter is the channel capacity in
        scaled-spend, scaled-target units.  The saturation is evaluated on
        size-normalized spend ``x / cap_c**rho`` in ``[0, 1]``, so the
        library's default priors are sensible for every campaign.
    data_vars : list[str]
        Single data variable in ``mmm.xarray_dataset`` holding campaign
        spend with dims ``("date", *mmm.dims, child_dim)``.
    prefix : str
        Prefix for all model variable names created by this effect.
    child_dim, parent_dim : str
        Names of the child and parent dimensions.  Default ``"campaign"``
        and ``"channel"``.
    tau_beta_sigma, tau_lam_sigma : float
        Scales of the HalfNormal priors on the pooling strength of the
        campaign-level amplitude and x-scale multipliers.  Smaller means
        stronger pooling.  Under the spend-share-weighted zero sum the
        prior sd of a campaign's log-multiplier is ``tau * sqrt(1 - u_c**2)``
        with ``u = share / |share|``, so it is not uniform across campaigns:
        the dominant campaign of a channel is pinned close to the channel
        mean (for a 90/5/3/2 split about 12x tighter than its siblings).
        This protects the channel total; ``tau_beta_sigma`` is therefore not
        "the campaign multiplier scale" for every campaign.
    rho : float
        Exponent tying each campaign's curve to its size (``cap_c``, the
        campaign's max channel-scaled spend).  At ``rho=1`` the campaign
        curve is the channel curve scaled by campaign size on both axes:
        split-invariant, with equal marginal returns at proportional spend.
    zero_sum_multipliers : bool
        When True (default), the campaign log-multipliers are constrained to
        a *spend-share-weighted* zero sum within each channel. This is a
        structural guarantee, not just a conditioning device: the campaign
        layer then preserves the spend-weighted geometric channel mean
        exactly, so introducing campaigns cannot move the channel-level
        contribution the design exists to protect (unweighted centring
        shifts it by double-digit percentages at lopsided spend splits). It
        also decorrelates the channel-level parameters from the dominant
        campaign's deviation, which is worth 6-9x ESS/sec at 90/5/3/2-style
        splits. Channels with a single campaign get multiplier 1 (fully
        pooled) automatically; so do campaigns with zero historical spend.
    covariate_var : str, optional
        Name of a variable in ``mmm.xarray_dataset`` with dims
        ``(child_dim, covariate_dim)`` holding per-campaign covariates
        (e.g. log clicks per unit spend, log impressions, CTR).  They enter
        the campaign amplitude multiplier and are spend-share-weighted-centred
        *within each channel* at build time, so they reallocate efficiency
        between a channel's campaigns without moving the channel-level total.
        A covariate is a hypothesis about efficiency, not evidence of it: its
        coefficient is learned only where the spend mix varies, and where it
        does not the coefficient stays at its prior, so the prior is where
        you state how far to trust the platform signal.  On a log scale
        (``log(clicks / spend)``) a coefficient of 1 reads "sales efficiency
        follows click efficiency one for one" and 0 "ignore it".
    covariate_dim : str
        Name of the covariate dimension.  Default ``"covariate"``.
    gamma_mu, gamma_sigma : float
        Mean and scale of the Normal prior on the covariate coefficients.
        The default ``gamma_mu=0`` states no direction; a positive mean with
        a small scale states trust in the covariate.
    """

    child_to_parent: dict[str, str]
    saturation: InstanceOf[SaturationTransformation] = Field(
        default_factory=MichaelisMentenSaturation
    )
    data_vars: list[str] = ["campaign_data"]
    prefix: str = "nested_media"
    child_dim: str = "campaign"
    parent_dim: str = "channel"
    tau_beta_sigma: float = 0.5
    tau_lam_sigma: float = 0.5
    rho: float = 1.0
    zero_sum_multipliers: bool = True
    covariate_var: str | None = None
    covariate_dim: str = "covariate"
    gamma_mu: float = 0.0
    gamma_sigma: float = 0.5

    model_config = {"arbitrary_types_allowed": True}

    def create_data(self, mmm: Model) -> None:
        """Register campaign spend plus the static index, scale and share data."""
        model = mmm.model
        p = self.prefix
        channel_coord = f"{p}_{self.parent_dim}"

        spend = mmm.xarray_dataset[self.data_vars[0]]
        campaigns = [str(c) for c in spend[self.child_dim].values]
        channels = self._validate_mapping(campaigns, model)
        channel_of = xr.DataArray(
            [self.child_to_parent[c] for c in campaigns],
            dims=self.child_dim,
            coords={self.child_dim: campaigns},
            name=channel_coord,
        )
        parent_idx = np.array([channels.index(ch) for ch in channel_of.values])

        # channel-level saturation priors, gathered to campaigns at build time
        if channel_coord not in model.coords:
            model.add_coord(channel_coord, channels)
        self.saturation = self.saturation.with_default_prior_dims((channel_coord,))
        self.saturation.prefix = f"{p}_saturation"
        super().create_data(mmm)

        scale, cap = _channel_scales(spend, channel_of, self.child_dim)
        scale = scale.sel({channel_coord: channels})
        dead = [c for c, k in zip(campaigns, cap.values, strict=True) if not k > 0]
        if dead:
            warnings.warn(
                f"Campaigns {dead} have no spend in the data. Their spend "
                "contributes nothing to the likelihood; their parameters are "
                "pinned to the pooled channel values (multiplier 1, cap "
                "fallback 1), so any curve read off them is prior-only.",
                UserWarning,
                stacklevel=2,
            )
        cap = cap.where(cap > 0, 1.0)
        total_spend = spend.sum([d for d in spend.dims if d != self.child_dim])
        share = _spend_shares(total_spend, channel_of, self.child_dim)

        pmd.Data(f"{p}_parent_idx", parent_idx, dims=(self.child_dim,))
        pmd.Data(
            f"{p}_parent_onehot",
            (parent_idx[:, None] == np.arange(len(channels))[None, :]).astype(float),
            dims=(self.child_dim, channel_coord),
        )
        pmd.Data(f"{p}_{self.parent_dim}_scale", scale.values, dims=(channel_coord,))
        pmd.Data(f"{p}_{self.child_dim}_cap", cap.values, dims=(self.child_dim,))
        pmd.Data(f"{p}_spend_share", share.values, dims=(self.child_dim,))

        if self.zero_sum_multipliers:
            self._register_zero_sum_blocks(model, campaigns, channel_of, share)

        if self.covariate_var is not None:
            cov = mmm.xarray_dataset[self.covariate_var]
            if set(cov.dims) != {self.child_dim, self.covariate_dim}:
                raise ValueError(
                    f"{self.covariate_var!r} must have dims exactly "
                    f"({self.child_dim!r}, {self.covariate_dim!r}); got {cov.dims}"
                )
            cov = cov.transpose(self.child_dim, self.covariate_dim)
            centred = _centred_covariates(cov, share, channel_of, self.child_dim)
            if self.covariate_dim not in model.coords:
                model.add_coord(
                    self.covariate_dim,
                    [str(c) for c in cov[self.covariate_dim].values]
                    if self.covariate_dim in cov.coords
                    else np.arange(cov.sizes[self.covariate_dim]),
                )
            pmd.Data(
                f"{p}_covariates",
                centred.transpose(self.child_dim, self.covariate_dim).values,
                dims=(self.child_dim, self.covariate_dim),
            )

    def _validate_mapping(self, campaigns: list[str], model: pm.Model) -> list[str]:
        """Check the campaign-to-channel map and return the channels in first-seen order."""
        missing = set(campaigns) - set(self.child_to_parent)
        extra = set(self.child_to_parent) - set(campaigns)
        if missing or extra:
            raise ValueError(
                "child_to_parent must cover exactly the campaigns in "
                f"{self.data_vars[0]!r}; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        channels = list(dict.fromkeys(self.child_to_parent[c] for c in campaigns))
        overlap = set(channels) & set(map(str, model.coords.get(self.parent_dim, ())))
        if overlap:
            warnings.warn(
                f"Channels {sorted(overlap)} are both decomposed into campaigns by "
                f"this effect and present in the model's {self.parent_dim!r} "
                "coordinate. The effect REPLACES the channel-level media term; "
                "keeping the channel in channel_columns double-counts its spend. "
                "Exclude decomposed channels from channel_columns.",
                UserWarning,
                stacklevel=3,
            )
        return channels

    def _parent_child_dim(self, parent: str) -> str:
        """Coordinate name of the live children of ``parent``."""
        return f"{self.prefix}_{parent}_{self.child_dim}"

    def _register_zero_sum_blocks(
        self,
        model: pm.Model,
        campaigns: list[str],
        channel_of: xr.DataArray,
        share: xr.DataArray,
    ) -> None:
        """Register one coordinate per constrained block and the scatter back to campaigns.

        A block is a channel's live campaigns (positive spend). Dead campaigns
        stay out so their multiplier is pinned at 1 instead of adding an
        unidentified free direction, and a channel needs two live campaigns
        to have any free direction at all. The blocks concatenate along one
        live-campaign coordinate; an index per campaign gathers them back to
        the campaign dimension, with dead campaigns pointing at a zero slot.
        """
        live = share.values > 0
        live_names: list[str] = []
        for channel in dict.fromkeys(channel_of.values):
            names = [
                c
                for c, ch, is_live in zip(
                    campaigns, channel_of.values, live, strict=True
                )
                if ch == channel and is_live
            ]
            if len(names) < 2:
                continue
            sub_dim = self._parent_child_dim(channel)
            if sub_dim not in model.coords:
                model.add_coord(sub_dim, names)
            live_names.extend(names)
        if not live_names:
            return
        live_dim = f"{self.prefix}_live_{self.child_dim}"
        if live_dim not in model.coords:
            model.add_coord(live_dim, live_names)
        # position of each campaign in the live vector; the slot after the
        # last live campaign holds a zero for campaigns outside every block
        position = {c: i for i, c in enumerate(live_names)}
        live_index = np.array([position.get(c, len(live_names)) for c in campaigns])
        pmd.Data(f"{self.prefix}_live_index", live_index, dims=(self.child_dim,))

    def _zero_sum_multiplier(self, model: pm.Model, name: str) -> XTensorVariable:
        """Standardised log-multiplier with a spend-share-weighted zero sum per channel.

        One :class:`~pymc_marketing.mmm.distributions.DimWeightedZeroSumNormal`
        per channel block, weighted by the block's spend shares, concatenated
        along the live-campaign coordinate and gathered back to the campaign
        dimension. Campaigns outside every block get zero, i.e. multiplier one;
        with no block at all every multiplier is pinned at one.
        """
        p = self.prefix
        live_dim = f"{p}_live_{self.child_dim}"
        if live_dim not in model.coords:
            return pmd.Deterministic(
                f"{p}_{name}", pmd.zeros_like(model[f"{p}_{self.child_dim}_cap"])
            )
        campaigns = [str(c) for c in model.coords[self.child_dim]]
        # spend shares are training constants, which is what the dims
        # distribution takes; the Data variable keeps them in the trace
        share = model[f"{p}_spend_share"].get_value()
        parts = []
        for channel in model.coords[f"{p}_{self.parent_dim}"]:
            sub_dim = self._parent_child_dim(channel)
            if sub_dim not in model.coords:
                continue
            idx = [campaigns.index(str(c)) for c in model.coords[sub_dim]]
            z = DimWeightedZeroSumNormal(
                f"{p}_{name}_{channel}", weights=share[idx], core_dims=sub_dim
            )
            parts.append(z.rename({sub_dim: live_dim}))
        zero_slot = pmd.zeros_like(parts[0].isel({live_dim: slice(0, 1)}))
        z_live = ptx.concat([*parts, zero_slot], dim=live_dim)
        gathered = z_live.isel({live_dim: model[f"{p}_live_index"]})
        return pmd.Deterministic(f"{p}_{name}", gathered)

    def set_data(self, mmm: Model, model: pm.Model, X: xr.Dataset) -> None:
        """Update ``campaign_data`` for a new prediction window.

        A wide DataFrame cannot carry the ``(date, campaign)`` spend, so the
        default MMM prediction route hands this effect a dataset without it.
        Silently reusing the training spend would be wrong, and a window of a
        different length would fail deep inside PyTensor. Instead the spend is
        set to zero over the new dates, with a warning: the campaigns then
        contribute nothing. To predict with campaign spend, pass an
        ``xr.Dataset`` ``X`` that carries the variable, as the budget
        optimizer's own dataset does.
        """
        var_name = self.data_vars[0]
        if var_name in X.data_vars:
            super().set_data(mmm, model, X)
            return
        current = model[var_name].get_value()
        dims = model.named_vars_to_dims[var_name]
        new_shape = tuple(
            X.sizes["date"] if dim == "date" else size
            for dim, size in zip(dims, current.shape, strict=True)
        )
        warnings.warn(
            f"{var_name!r} is not in the prediction data (a DataFrame cannot carry "
            "it), so campaign spend is set to zero over the new dates and the "
            "campaigns contribute nothing. Pass an xr.Dataset X with "
            f"{var_name!r} to predict with campaign spend.",
            UserWarning,
            stacklevel=2,
        )
        pm.set_data({var_name: np.zeros(new_shape, dtype=current.dtype)}, model=model)

    def create_effect(self, mmm: Model) -> XTensorVariable:
        """Build the nested campaign media contribution."""
        model = mmm.model
        p = self.prefix
        channel_coord_name = f"{p}_{self.parent_dim}"

        x = model[self.data_vars[0]]
        parent_idx = model[f"{p}_parent_idx"]
        onehot = model[f"{p}_parent_onehot"]
        scale = model[f"{p}_{self.parent_dim}_scale"]
        cap = model[f"{p}_{self.child_dim}_cap"]

        def gather(var):
            return var[{channel_coord_name: parent_idx}]

        x_scaled = x / gather(scale)

        tau_beta = pmd.HalfNormal(f"{p}_tau_beta", sigma=self.tau_beta_sigma)
        tau_lam = pmd.HalfNormal(f"{p}_tau_lam", sigma=self.tau_lam_sigma)
        if self.zero_sum_multipliers:
            z_beta = self._zero_sum_multiplier(model, "z_beta")
            z_lam = self._zero_sum_multiplier(model, "z_lam")
        else:
            z_beta = pmd.Normal(f"{p}_z_beta", 0.0, 1.0, dims=(self.child_dim,))
            z_lam = pmd.Normal(f"{p}_z_lam", 0.0, 1.0, dims=(self.child_dim,))

        log_mult = tau_beta * z_beta
        if self.covariate_var is not None:
            cov = model[f"{p}_covariates"]
            gamma = pmd.Normal(
                f"{p}_gamma",
                self.gamma_mu,
                self.gamma_sigma,
                dims=(self.covariate_dim,),
            )
            log_mult = log_mult + (cov * gamma).sum(dim=self.covariate_dim)
        beta_multiplier = pmd.Deterministic(
            f"{p}_beta_multiplier", pmd.math.exp(log_mult)
        )
        lam_multiplier = pmd.Deterministic(
            f"{p}_lam_multiplier", pmd.math.exp(tau_lam * z_lam)
        )

        # channel-level saturation parameters, gathered to campaigns
        shape_params = {
            name: gather(var)
            for name, var in self.saturation._create_distributions().items()
        }

        # the campaign curve is the channel curve scaled by campaign size on
        # both axes: split-invariant for any saturation shape
        size = cap**self.rho
        x_rel = x_scaled / (size * lam_multiplier)
        curve = self.saturation.function(x_rel, dim="date", **shape_params)
        campaign_contribution = pmd.Deterministic(
            f"{p}_{self.child_dim}_contribution", size * beta_multiplier * curve
        )

        amplitude = _AMPLITUDE_PARAM.get(type(self.saturation).__name__)
        if amplitude is not None:
            pmd.Deterministic(
                f"{p}_beta_{self.child_dim}",
                gather(model[self.saturation.variable_mapping[amplitude]])
                * size
                * beta_multiplier,
            )
        if isinstance(self.saturation, MichaelisMentenSaturation):
            # half-saturation point of campaign c on the scaled-spend axis
            pmd.Deterministic(
                f"{p}_lam_{self.child_dim}",
                gather(model[self.saturation.variable_mapping["lam"]])
                * size
                * lam_multiplier,
            )

        pmd.Deterministic(
            f"{p}_{self.parent_dim}_contribution",
            (campaign_contribution * onehot).sum(dim=self.child_dim),
        )
        return pmd.Deterministic(
            f"{p}_effect_contribution",
            campaign_contribution.sum(dim=self.child_dim),
        )

    def add_lift_test_measurements(
        self,
        df_lift_test: pd.DataFrame,
        mmm: Model,
        dist: Callable[..., XTensorVariable] = lognormal_relative_lift,
        name: str | None = None,
        target_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> "NestedMediaEffect":
        """Calibrate campaign saturation curves with lift-test results.

        For each row, the model's estimated lift on the campaign's own curve,
        ``cap**rho * mult * (S(x + delta_x) - S(x))``, is conditioned on the
        measured ``delta_y`` with ``dist``. Works for any saturation shape.
        This is the strongest campaign-level identification source: it does
        not rely on flighting contrasts in the historical spend.

        Parameters
        ----------
        df_lift_test : pd.DataFrame
            One row per lift test with columns ``{child_dim}``, ``x``,
            ``delta_x``, ``delta_y``, ``sigma``. ``x`` and ``delta_x`` are in
            spend units; they are scaled internally by the campaign's channel
            scale. ``delta_y`` and ``sigma`` are in target units.
        mmm : Model
            The MMM the effect was built into. The model must be built.
        dist : callable, optional
            Likelihood for the lift measurements, called with ``name``,
            ``mu`` (the estimated lift), ``sigma`` and ``observed``. By
            default :func:`lognormal_relative_lift`, a positive likelihood
            with the noise as relative error on the measurement. A
            ``pmd.Gamma`` or a moment-matched LogNormal is not recommended:
            they barely penalise an estimated lift near zero, which gives
            the posterior a degenerate mode with a collapsed half-saturation.
        name : str, optional
            Name of the likelihood, defaults to
            ``f"{prefix}_lift_measurements"``.
        target_transform : Callable, optional
            Function ``(n, 1) -> (n, 1)`` scaling ``delta_y``/``sigma`` into
            the model's (scaled) target units. Defaults to dividing by
            ``mmm.scalers._target`` when available (per-dim aware; model dim
            columns are then required in ``df_lift_test``), otherwise the
            identity (targets assumed unscaled).
        """
        model = mmm.model
        p = self.prefix
        channel_coord_name = f"{p}_{self.parent_dim}"
        if f"{p}_beta_multiplier" not in model.named_vars:
            raise RuntimeError(
                "The model has not been built yet. Build the model before "
                "adding lift test measurements."
            )
        required = {self.child_dim, "x", "delta_x", "delta_y", "sigma"}
        missing = required - set(df_lift_test.columns)
        if missing:
            raise KeyError(f"df_lift_test is missing columns {sorted(missing)}")

        campaigns = [str(c) for c in model.coords[self.child_dim]]
        unknown = set(df_lift_test[self.child_dim].astype(str)) - set(campaigns)
        if unknown:
            raise ValueError(f"Unknown campaigns in df_lift_test: {sorted(unknown)}")
        _check_lift_rows(df_lift_test)

        rows = df_lift_test.reset_index(drop=True)
        row_channel = rows[self.child_dim].astype(str).map(self.child_to_parent)
        channel_scale = xr.DataArray(
            model[f"{p}_{self.parent_dim}_scale"].get_value(),
            dims=channel_coord_name,
            coords={channel_coord_name: list(model.coords[channel_coord_name])},
        )
        row_scale = channel_scale.sel(
            {channel_coord_name: xr.DataArray(row_channel.to_numpy(), dims="row")}
        ).to_numpy()

        if target_transform is None:
            # Scale delta_y and sigma into the model's target units. The
            # fitted target scaler is a scalar, or an array over some of the
            # model dims: select on exactly the dims it carries, which the
            # lift table must then provide as columns.
            target_scale = getattr(getattr(mmm, "scalers", None), "_target", None)
            if target_scale is None:
                row_target_scale = np.ones(len(rows))
            else:
                target_scale = xr.DataArray(target_scale)
                missing_dims = set(target_scale.dims) - set(rows.columns)
                if missing_dims:
                    raise KeyError(
                        f"df_lift_test is missing the model dim columns "
                        f"{sorted(missing_dims)} needed to scale delta_y/sigma"
                    )
                row_target_scale = np.broadcast_to(
                    target_scale.sel(
                        {
                            d: xr.DataArray(rows[d].to_numpy(), dims="row")
                            for d in target_scale.dims
                        }
                    ).to_numpy(),
                    (len(rows),),
                )

            def target_transform(values: np.ndarray) -> np.ndarray:
                return values / row_target_scale[:, None]

        def scale_target(col: pd.Series) -> np.ndarray:
            return target_transform(col.to_numpy()[:, None])[:, 0]

        df_scaled = rows.assign(
            **{
                "x": rows["x"] / row_scale,
                "delta_x": rows["delta_x"] / row_scale,
                "delta_y": scale_target(rows["delta_y"]),
                "sigma": scale_target(rows["sigma"]),
                # the saturation's channel-level params are indexed by the
                # effect's channel coordinate
                channel_coord_name: row_channel,
            }
        )

        rho = self.rho
        saturation_function = self.saturation.function

        def campaign_curve(x, cap, beta_mult, lam_mult, **shape_params):
            size = cap**rho
            return (
                size
                * beta_mult
                * saturation_function(x / (size * lam_mult), **shape_params)
            )

        variable_mapping = {
            "cap": f"{p}_{self.child_dim}_cap",
            "beta_mult": f"{p}_beta_multiplier",
            "lam_mult": f"{p}_lam_multiplier",
            **self.saturation.variable_mapping,
        }

        add_saturation_observations(
            df_scaled,
            variable_mapping=variable_mapping,
            saturation_function=campaign_curve,
            model=model,
            dist=dist,
            name=name or f"{p}_lift_measurements",
        )
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict."""
        data = self.model_dump(mode="json", exclude={"saturation"})
        data["saturation"] = self.saturation.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NestedMediaEffect":
        """Reconstruct from a dict."""
        work = {k: v for k, v in data.items() if k != "__type__"}
        if isinstance(work.get("saturation"), dict):
            work["saturation"] = serialization.deserialize(work["saturation"])
        return cls(**work)
